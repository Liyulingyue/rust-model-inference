//! SenseNova-U1.6 `language_model` trunk — the dual-branch Qwen3 decoder.
//!
//! Every attention and MLP weight of the checkpoint exists twice: a plain copy
//! used by the *understanding* branch (text prefix) and a `*_mot_gen` copy used
//! by the *generation* branch (image tokens). Both branches share one geometry
//! and one M-RoPE layout; they differ only in which weight set is read and in
//! the attention mask (causal vs. block-bidirectional).
//!
//! Per head, the 128 rotary dims are three unequal segments, each rotated with
//! its own position and frequency base:
//!
//! | segment | dims      | position          | theta |
//! | ------- | --------- | ----------------- | ----- |
//! | t       | `0..64`   | token / timestep  | 5e6   |
//! | h       | `64..96`  | image row         | 1e4   |
//! | w       | `96..128` | image column      | 1e4   |
//!
//! Each segment is a plain half-split (NeoX) rotation. Upstream builds three
//! independent `Qwen3RotaryEmbedding` tables through
//! `_rope_init_fn_keep_freq_range`, which computes `inv_freq` for `head_dim * 2`
//! and then keeps every second entry:
//!
//! ```text
//! dim2 = 2 * seg_dim;  full[i] = theta^(-i/dim2) for i in 0..dim2/2
//! kept[j] = full[2j]  = theta^(-2j/dim2) = theta^(-j/seg_dim)
//! ```
//!
//! i.e. exactly the standard `rope_neox(head_dim = seg_dim, theta)` frequency
//! ladder. The t slice is therefore `rope_neox(64, 5e6)` and each h/w slice is
//! `rope_neox(32, 1e4)`, which is what [`Dims::rope_scales`] encodes.
//!
//! See `docs/develop/SENSENOVA_U1.md` §3 for the contract and
//! `references/src/sensenova_u1/models/neo_unify/modeling_qwen3.py`
//! (`Qwen3Attention::forward_und` / `forward_gen`, `Qwen3DecoderLayer`,
//! `Qwen3Model::forward`) for the upstream source this mirrors.
//!
//! [`Trunk`] borrows no tensor bytes: it caches only metadata, and every
//! forward call re-wraps the mmap-backed slices of the caller's
//! [`TensorSource`] into [`Weight`]s. That keeps the trunk free of lifetimes so
//! the sampler can hold it beside its own sources.

use super::config::{
    supported_linear_types, FFN, HEAD_DIM, HIDDEN, KV_HEADS, LAYERS, QUERY_HEADS, RMS_EPS,
    ROPE_THETA_HW, ROPE_THETA_T, VOCAB,
};
use crate::core::tensor::{require_tensor_meta, GGMLType, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::models::qwen35::trunk::weights::{load_weight, load_weight_f32};
use crate::ops::attention_value::attention_value_reduce;
use crate::ops::embedding::embedding_lookup;
use crate::ops::kernel::{PreparedRows, Weight};
use crate::ops::rope::neox::rope_neox_inplace_scalar;
use crate::ops::{
    dot_f32, rms_norm_inplace, rope_sin_cos, silu_inplace, softmax_inplace, vec_add_into,
    vec_mul_inplace,
};

// =============================================================================
// Public surface
// =============================================================================

/// Per-layer KV of one prefix pass.
///
/// Layout per layer: `k`/`v` are `[KV_HEADS, seq_len, HEAD_DIM]` row-major
/// (head-major, as the attention loop wants them).
pub(crate) struct PrefixCache {
    pub(crate) len: usize,
    pub(crate) layers: Vec<LayerKv>,
}

pub(crate) struct LayerKv {
    pub(crate) k: Vec<f32>,
    pub(crate) v: Vec<f32>,
}

/// The Qwen3-style dual-branch trunk. Holds geometry and per-layer GGUF facts
/// only — no borrowed bytes, no weights.
pub(crate) struct Trunk {
    dims: Dims,
    layers: Vec<LayerMeta>,
}

impl Trunk {
    /// Validate + cache tensor metadata. Reads no payload beyond what
    /// [`TensorSource::tensor_info`] returns.
    pub(crate) fn new(source: &dyn TensorSource) -> Result<Self, String> {
        Self::build(source, Dims::contract())
    }

    /// Understanding-branch forward over text token ids.
    ///
    /// Returns the KV cache plus the final hidden state `[seq_len, HIDDEN]`
    /// (post `norm`). Positions are the text M-RoPE indexes `(t, 0, 0)` with
    /// `t = 0..seq_len`; the mask is causal, which is what upstream's
    /// `create_block_causal_mask(t_index)` degenerates to for a strictly
    /// increasing text prefix (contract §3.2/§3.3).
    pub(crate) fn encode_prefix(
        &self,
        source: &dyn TensorSource,
        tokens: &[u32],
        pool: &ComputePool,
    ) -> Result<(PrefixCache, Vec<f32>), String> {
        let dims = self.dims;
        let rows = tokens.len();
        if rows == 0 {
            return Err("SenseNova-U1.6 encode_prefix needs at least one token".into());
        }
        if let Some((index, token)) = tokens
            .iter()
            .enumerate()
            .find(|(_, token)| **token as usize >= dims.vocab)
        {
            return Err(format!(
                "SenseNova-U1.6 token id {token} at position {index} is outside the \
                 {}-entry vocabulary",
                dims.vocab
            ));
        }

        const EMBED: &str = "language_model.model.embed_tokens.weight";
        let embed = source
            .tensor_slice(EMBED)
            .ok_or_else(|| format!("Missing tensor data: {EMBED}"))?;
        let needed = dims
            .vocab
            .checked_mul(dims.hidden)
            .and_then(|n| n.checked_mul(4))
            .ok_or("SenseNova-U1.6 embedding table size overflow")?;
        if embed.len() < needed {
            return Err(format!(
                "SenseNova-U1.6 embedding table is {} bytes, expected {needed}",
                embed.len()
            ));
        }

        let mut scratch = Scratch::new(dims, rows);
        for (row, &token) in tokens.iter().enumerate() {
            let range = row * dims.hidden..(row + 1) * dims.hidden;
            embedding_lookup(
                embed,
                token,
                dims.hidden,
                GGMLType::F32,
                &mut scratch.x[range],
            );
        }

        // Text prefix: t = 0..L-1, h = w = 0 (contract §3.3).
        let positions: Vec<[usize; 3]> = (0..rows).map(|t| [t, 0, 0]).collect();
        let mut cache = PrefixCache {
            len: rows,
            layers: Vec::with_capacity(dims.layers),
        };
        for layer in 0..dims.layers {
            let run = LayerRun {
                branch: Branch::Und,
                rows,
                positions: &positions,
                prefix: None,
                causal: true,
            };
            self.run_layer(source, layer, &run, &mut scratch, pool)?;
            cache
                .layers
                .push(store_kv(&scratch.k, &scratch.v, rows, dims));
        }

        let norm = self.load_norm(source, "language_model.model.norm.weight")?;
        for row in scratch.x.chunks_exact_mut(dims.hidden) {
            rms_norm_inplace(row, &norm, dims.eps);
        }
        Ok((cache, scratch.x))
    }

    /// Generation-branch forward over `image_embeds` `[n_tokens, HIDDEN]`
    /// (already including the timestep + noise-scale embeddings).
    ///
    /// Positions are M-RoPE: `t = text_len` (constant), `h = idx / token_w`,
    /// `w = idx % token_w`. Attends to `cache` (prefix, und-branch K/V) plus
    /// the current block **bidirectionally**. Returns hidden states
    /// `[n_tokens, HIDDEN]` after `norm_mot_gen`. Never mutates `cache`.
    pub(crate) fn denoise(
        &self,
        source: &dyn TensorSource,
        image_embeds: &[f32],
        text_len: usize,
        token_w: usize,
        cache: &PrefixCache,
        pool: &ComputePool,
    ) -> Result<Vec<f32>, String> {
        let dims = self.dims;
        if image_embeds.is_empty() {
            return Err("SenseNova-U1.6 denoise needs at least one image token".into());
        }
        if !image_embeds.len().is_multiple_of(dims.hidden) {
            return Err(format!(
                "SenseNova-U1.6 image_embeds length {} is not a multiple of HIDDEN ({})",
                image_embeds.len(),
                dims.hidden
            ));
        }
        let rows = image_embeds.len() / dims.hidden;
        if token_w == 0 {
            return Err("SenseNova-U1.6 denoise requires token_w > 0".into());
        }
        if !rows.is_multiple_of(token_w) {
            return Err(format!(
                "SenseNova-U1.6 denoise got {rows} image tokens, which is not a whole \
                 number of rows of token_w = {token_w}"
            ));
        }
        if cache.len != text_len {
            return Err(format!(
                "SenseNova-U1.6 prefix cache holds {} tokens but text_len is {text_len}",
                cache.len
            ));
        }
        if cache.layers.len() != dims.layers {
            return Err(format!(
                "SenseNova-U1.6 prefix cache has {} layers, expected {}",
                cache.layers.len(),
                dims.layers
            ));
        }
        let kv_span = dims
            .kv_heads
            .checked_mul(cache.len)
            .and_then(|n| n.checked_mul(dims.head_dim))
            .ok_or("SenseNova-U1.6 prefix cache size overflow")?;
        for (layer, kv) in cache.layers.iter().enumerate() {
            if kv.k.len() != kv_span || kv.v.len() != kv_span {
                return Err(format!(
                    "SenseNova-U1.6 prefix cache layer {layer} holds {} / {} floats, \
                     expected {kv_span}",
                    kv.k.len(),
                    kv.v.len()
                ));
            }
        }

        let positions: Vec<[usize; 3]> = (0..rows)
            .map(|index| [text_len, index / token_w, index % token_w])
            .collect();

        let mut scratch = Scratch::new(dims, rows);
        scratch.x.copy_from_slice(image_embeds);
        for layer in 0..dims.layers {
            let kv = &cache.layers[layer];
            let run = LayerRun {
                branch: Branch::Gen,
                rows,
                positions: &positions,
                prefix: Some((&kv.k, &kv.v, cache.len)),
                causal: false,
            };
            self.run_layer(source, layer, &run, &mut scratch, pool)?;
        }

        let norm = self.load_norm(source, "language_model.model.norm_mot_gen.weight")?;
        for row in scratch.x.chunks_exact_mut(dims.hidden) {
            rms_norm_inplace(row, &norm, dims.eps);
        }
        Ok(scratch.x)
    }
}

// =============================================================================
// Geometry
// =============================================================================

/// Every shape the trunk derives. [`Dims::contract`] is the real
/// SenseNova-U1.6-8B geometry from `config.rs`; the unit tests build a smaller
/// one so the same code path can run over a few kilobytes of synthetic weights.
#[derive(Clone, Copy, Debug)]
struct Dims {
    hidden: usize,
    layers: usize,
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    ffn: usize,
    vocab: usize,
    eps: f32,
    /// `theta^(-2/seg_dim)` for the t, h and w segments, in that order. This is
    /// the ratio between two consecutive `inv_freq` entries, so the segment's
    /// frequency ladder is one running product — see the module docs for why
    /// this equals upstream's `_rope_init_fn_keep_freq_range`.
    rope_scales: [f32; 3],
}

impl Dims {
    fn build(
        hidden: usize,
        layers: usize,
        query_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        ffn: usize,
        vocab: usize,
    ) -> Self {
        let rope_t = head_dim / 2;
        let rope_hw = head_dim / 4;
        let scale_t = ROPE_THETA_T.powf(-2.0 / rope_t as f32);
        let scale_hw = ROPE_THETA_HW.powf(-2.0 / rope_hw as f32);
        Self {
            hidden,
            layers,
            query_heads,
            kv_heads,
            head_dim,
            ffn,
            vocab,
            eps: RMS_EPS,
            rope_scales: [scale_t, scale_hw, scale_hw],
        }
    }

    fn contract() -> Self {
        Self::build(HIDDEN, LAYERS, QUERY_HEADS, KV_HEADS, HEAD_DIM, FFN, VOCAB)
    }

    /// Width of the time-axis slice (`head_dim / 2`).
    fn rope_t(&self) -> usize {
        self.head_dim / 2
    }

    /// Width of each h/w slice (`head_dim / 4`).
    fn rope_hw(&self) -> usize {
        self.head_dim / 4
    }

    fn q_dim(&self) -> usize {
        self.query_heads * self.head_dim
    }

    fn kv_dim(&self) -> usize {
        self.kv_heads * self.head_dim
    }

    /// Query heads per KV head.
    fn group(&self) -> usize {
        self.query_heads / self.kv_heads
    }

    /// `(offset, width, inv_freq ratio)` of the t, h and w slices, in head
    /// order. The rotation offsets are `0`, `rope_t()` and `rope_t() + rope_hw()`.
    fn rope_segments(&self) -> [(usize, usize, f32); 3] {
        let t = self.rope_t();
        let hw = self.rope_hw();
        [
            (0, t, self.rope_scales[0]),
            (t, hw, self.rope_scales[1]),
            (t + hw, hw, self.rope_scales[2]),
        ]
    }

    fn check(&self) -> Result<(), String> {
        if self.hidden == 0
            || self.layers == 0
            || self.head_dim == 0
            || self.ffn == 0
            || self.vocab == 0
        {
            return Err("SenseNova-U1.6 trunk geometry has a zero dimension".into());
        }
        if !self.head_dim.is_multiple_of(4) {
            return Err(format!(
                "SenseNova-U1.6 head_dim {} must be a multiple of 4 (64 + 32 + 32 M-RoPE)",
                self.head_dim
            ));
        }
        if self.kv_heads == 0 || !self.query_heads.is_multiple_of(self.kv_heads) {
            return Err(format!(
                "SenseNova-U1.6 needs query_heads ({}) to be a multiple of kv_heads ({})",
                self.query_heads, self.kv_heads
            ));
        }
        if self.q_dim() != self.hidden {
            return Err(format!(
                "SenseNova-U1.6 q_proj output ({}) must equal HIDDEN ({})",
                self.q_dim(),
                self.hidden
            ));
        }
        Ok(())
    }
}

/// Which half of the MoT weight set a layer pass reads.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Branch {
    /// Understanding / text prefix. Plain weight names.
    Und,
    /// Generation / image tokens. `*_mot_gen` weight names.
    Gen,
}

impl Branch {
    fn suffix(self) -> &'static str {
        match self {
            Self::Und => "",
            Self::Gen => "_mot_gen",
        }
    }
}

// =============================================================================
// Metadata validation
// =============================================================================

/// Per-layer GGUF facts cached by [`Trunk::build`] so the forward path never
/// re-derives them. The two flags per group say which activation format the
/// [`PreparedRows`] scratch has to hold before the matmul: Q8_0 for the
/// `Q4_0`/`Q8_0` kernels, Q8_K for the K-quants.
#[derive(Default, Clone, Copy)]
struct LayerMeta {
    qkv_q8: bool,
    qkv_q8k: bool,
    o_q8: bool,
    o_q8k: bool,
    mlp_q8: bool,
    mlp_q8k: bool,
    down_q8: bool,
    down_q8k: bool,
}

impl LayerMeta {
    fn note(&mut self, slot: Projection, ggml_type: GGMLType) {
        let needs_q8 = matches!(
            ggml_type,
            GGMLType::Q4_0
                | GGMLType::Q4_1
                | GGMLType::Q5_0
                | GGMLType::Q5_1
                | GGMLType::Q8_0
                | GGMLType::Q1_0
        );
        let needs_q8k = matches!(
            ggml_type,
            GGMLType::Q2K
                | GGMLType::Q3K
                | GGMLType::Q4K
                | GGMLType::Q5K
                | GGMLType::Q6K
                | GGMLType::IQ1_S
                | GGMLType::IQ1_M
                | GGMLType::IQ2_XXS
                | GGMLType::IQ2_XS
                | GGMLType::IQ2_S
                | GGMLType::IQ3_XXS
                | GGMLType::IQ3_S
                | GGMLType::IQ4_NL
                | GGMLType::IQ4_XS
        );
        let (q8, q8k) = match slot {
            Projection::Qkv => (&mut self.qkv_q8, &mut self.qkv_q8k),
            Projection::O => (&mut self.o_q8, &mut self.o_q8k),
            Projection::Mlp => (&mut self.mlp_q8, &mut self.mlp_q8k),
            Projection::Down => (&mut self.down_q8, &mut self.down_q8k),
        };
        *q8 |= needs_q8;
        *q8k |= needs_q8k;
    }
}

#[derive(Clone, Copy)]
enum Projection {
    Qkv,
    O,
    Mlp,
    Down,
}

/// Tensor names of one decoder layer, for one branch.
struct LayerNames {
    input_norm: String,
    post_norm: String,
    q_proj: String,
    k_proj: String,
    v_proj: String,
    o_proj: String,
    q_norm: String,
    q_norm_hw: String,
    k_norm: String,
    k_norm_hw: String,
    gate: String,
    up: String,
    down: String,
}

impl LayerNames {
    fn new(layer: usize, branch: Branch) -> Self {
        let base = format!("language_model.model.layers.{layer}");
        let suffix = branch.suffix();
        Self {
            input_norm: format!("{base}.input_layernorm{suffix}.weight"),
            post_norm: format!("{base}.post_attention_layernorm{suffix}.weight"),
            q_proj: format!("{base}.self_attn.q_proj{suffix}.weight"),
            k_proj: format!("{base}.self_attn.k_proj{suffix}.weight"),
            v_proj: format!("{base}.self_attn.v_proj{suffix}.weight"),
            o_proj: format!("{base}.self_attn.o_proj{suffix}.weight"),
            q_norm: format!("{base}.self_attn.q_norm{suffix}.weight"),
            q_norm_hw: format!("{base}.self_attn.q_norm_hw{suffix}.weight"),
            k_norm: format!("{base}.self_attn.k_norm{suffix}.weight"),
            k_norm_hw: format!("{base}.self_attn.k_norm_hw{suffix}.weight"),
            gate: format!("{base}.mlp{suffix}.gate_proj.weight"),
            up: format!("{base}.mlp{suffix}.up_proj.weight"),
            down: format!("{base}.mlp{suffix}.down_proj.weight"),
        }
    }
}

impl Trunk {
    fn build(source: &dyn TensorSource, dims: Dims) -> Result<Self, String> {
        dims.check()?;
        let linear = supported_linear_types();
        let f32_only = [GGMLType::F32];

        let hidden = dims.hidden as u64;
        let q_dim = dims.q_dim() as u64;
        let kv_dim = dims.kv_dim() as u64;
        let ffn = dims.ffn as u64;
        let norm_dim = dims.rope_t() as u64;

        require_tensor_meta(
            source,
            "language_model.model.embed_tokens.weight",
            &[hidden, dims.vocab as u64],
            &f32_only,
        )?;
        for name in [
            "language_model.model.norm.weight",
            "language_model.model.norm_mot_gen.weight",
        ] {
            require_tensor_meta(source, name, &[hidden], &f32_only)?;
        }

        let mut layers = Vec::with_capacity(dims.layers);
        for layer in 0..dims.layers {
            let mut meta = LayerMeta::default();
            for branch in [Branch::Und, Branch::Gen] {
                let names = LayerNames::new(layer, branch);
                require_tensor_meta(source, &names.input_norm, &[hidden], &f32_only)?;
                require_tensor_meta(source, &names.post_norm, &[hidden], &f32_only)?;
                // q_proj is `[HIDDEN, QUERY_HEADS * HEAD_DIM]`; k/v are
                // `[HIDDEN, KV_HEADS * HEAD_DIM]`.
                for (name, out) in [
                    (&names.q_proj, q_dim),
                    (&names.k_proj, kv_dim),
                    (&names.v_proj, kv_dim),
                ] {
                    let ggml_type = require_tensor_meta(source, name, &[hidden, out], linear)?;
                    meta.note(Projection::Qkv, ggml_type);
                }
                let ggml_type =
                    require_tensor_meta(source, &names.o_proj, &[q_dim, hidden], linear)?;
                meta.note(Projection::O, ggml_type);
                for name in [
                    &names.q_norm,
                    &names.q_norm_hw,
                    &names.k_norm,
                    &names.k_norm_hw,
                ] {
                    require_tensor_meta(source, name, &[norm_dim], &f32_only)?;
                }
                for name in [&names.gate, &names.up] {
                    let ggml_type = require_tensor_meta(source, name, &[hidden, ffn], linear)?;
                    meta.note(Projection::Mlp, ggml_type);
                }
                let ggml_type = require_tensor_meta(source, &names.down, &[ffn, hidden], linear)?;
                meta.note(Projection::Down, ggml_type);
            }
            layers.push(meta);
        }

        // A checkpoint with more layers than the pinned contract would be
        // silently truncated by the loop above, so reject it explicitly.
        if source
            .tensor_info(&format!(
                "language_model.model.layers.{}.self_attn.q_proj.weight",
                dims.layers
            ))
            .is_some()
        {
            return Err(format!(
                "SenseNova-U1.6 expects {} decoder layers",
                dims.layers
            ));
        }

        Ok(Self { dims, layers })
    }

    fn load_norm(&self, source: &dyn TensorSource, name: &str) -> Result<Vec<f32>, String> {
        let weight = load_weight_f32(source, name)
            .ok_or_else(|| format!("Missing or unreadable tensor: {name}"))?;
        if weight.len() != self.dims.hidden {
            return Err(format!(
                "Tensor {name} holds {} values, expected {}",
                weight.len(),
                self.dims.hidden
            ));
        }
        Ok(weight)
    }
}

// =============================================================================
// One layer
// =============================================================================

/// The four per-head RMSNorm weights of a branch.
struct HeadNorms {
    q_t: Vec<f32>,
    q_hw: Vec<f32>,
    k_t: Vec<f32>,
    k_hw: Vec<f32>,
}

/// All weights of one decoder layer for one branch.
struct LayerWeights<'a> {
    input_norm: Vec<f32>,
    post_norm: Vec<f32>,
    q_proj: Weight<'a>,
    k_proj: Weight<'a>,
    v_proj: Weight<'a>,
    o_proj: Weight<'a>,
    norms: HeadNorms,
    gate: Weight<'a>,
    up: Weight<'a>,
    down: Weight<'a>,
}

fn load_layer_weights<'a>(
    source: &'a dyn TensorSource,
    layer: usize,
    branch: Branch,
    dims: Dims,
) -> Result<LayerWeights<'a>, String> {
    let names = LayerNames::new(layer, branch);
    let weight = |name: &str| -> Result<Weight<'a>, String> {
        load_weight(source, name).ok_or_else(|| format!("Missing or unreadable tensor: {name}"))
    };
    let f32_values = |name: &str, width: usize| -> Result<Vec<f32>, String> {
        let values = load_weight_f32(source, name)
            .ok_or_else(|| format!("Missing or unreadable tensor: {name}"))?;
        if values.len() != width {
            return Err(format!(
                "Tensor {name} holds {} values, expected {width}",
                values.len()
            ));
        }
        Ok(values)
    };
    let norm_dim = dims.rope_t();
    Ok(LayerWeights {
        input_norm: f32_values(&names.input_norm, dims.hidden)?,
        post_norm: f32_values(&names.post_norm, dims.hidden)?,
        q_proj: weight(&names.q_proj)?,
        k_proj: weight(&names.k_proj)?,
        v_proj: weight(&names.v_proj)?,
        o_proj: weight(&names.o_proj)?,
        norms: HeadNorms {
            q_t: f32_values(&names.q_norm, norm_dim)?,
            q_hw: f32_values(&names.q_norm_hw, norm_dim)?,
            k_t: f32_values(&names.k_norm, norm_dim)?,
            k_hw: f32_values(&names.k_norm_hw, norm_dim)?,
        },
        gate: weight(&names.gate)?,
        up: weight(&names.up)?,
        down: weight(&names.down)?,
    })
}

/// One pass over one decoder layer.
struct LayerRun<'a> {
    branch: Branch,
    rows: usize,
    /// M-RoPE `(t, h, w)` per row.
    positions: &'a [[usize; 3]],
    /// `(k, v, len)` of the prefix cache, head-major. `None` for the prefix
    /// pass itself.
    prefix: Option<(&'a [f32], &'a [f32], usize)>,
    causal: bool,
}

impl Trunk {
    fn run_layer(
        &self,
        source: &dyn TensorSource,
        layer: usize,
        run: &LayerRun<'_>,
        scratch: &mut Scratch,
        pool: &ComputePool,
    ) -> Result<(), String> {
        let dims = self.dims;
        let rows = run.rows;
        let meta = self.layers[layer];
        let weights = load_layer_weights(source, layer, run.branch, dims)?;
        let context = |what: &str| format!("SenseNova-U1.6 layer {layer} {what}");

        // ---- attention -----------------------------------------------------
        for (x_row, normed_row) in scratch
            .x
            .chunks_exact(dims.hidden)
            .zip(scratch.normed.chunks_exact_mut(dims.hidden))
        {
            normed_row.copy_from_slice(x_row);
            rms_norm_inplace(normed_row, &weights.input_norm, dims.eps);
        }
        scratch
            .prepared
            .prepare(
                &scratch.normed,
                rows,
                dims.hidden,
                meta.qkv_q8,
                meta.qkv_q8k,
            )
            .map_err(|error| context(&error))?;
        scratch
            .prepared
            .matmul_group(
                &scratch.normed,
                [
                    (&weights.q_proj, &mut scratch.q[..rows * dims.q_dim()]),
                    (&weights.k_proj, &mut scratch.k[..rows * dims.kv_dim()]),
                    (&weights.v_proj, &mut scratch.v[..rows * dims.kv_dim()]),
                ],
                pool,
            )
            .map_err(|error| context(&error))?;

        // Per-head RMSNorm on the 64-wide t half and the 64-wide hw half
        // *before* the h/w split, then the three-segment rotation.
        head_norm_and_rope(
            &mut scratch.q,
            rows,
            dims.query_heads,
            dims,
            &weights.norms.q_t,
            &weights.norms.q_hw,
            run.positions,
            &mut scratch.cos,
            &mut scratch.sin,
        );
        head_norm_and_rope(
            &mut scratch.k,
            rows,
            dims.kv_heads,
            dims,
            &weights.norms.k_t,
            &weights.norms.k_hw,
            run.positions,
            &mut scratch.cos,
            &mut scratch.sin,
        );

        attend(
            &scratch.q,
            &scratch.k,
            &scratch.v,
            run.prefix,
            &mut scratch.attn,
            rows,
            run.causal,
            dims,
            pool,
        );

        scratch
            .prepared
            .prepare(&scratch.attn, rows, dims.q_dim(), meta.o_q8, meta.o_q8k)
            .map_err(|error| context(&error))?;
        scratch
            .prepared
            .matmul(
                &weights.o_proj,
                &scratch.attn,
                &mut scratch.proj[..rows * dims.hidden],
                pool,
            )
            .map_err(|error| context(&error))?;
        for (projected, x_row) in scratch
            .proj
            .chunks_exact(dims.hidden)
            .zip(scratch.x.chunks_exact_mut(dims.hidden))
        {
            vec_add_into(projected, x_row);
        }

        // ---- SwiGLU FFN ----------------------------------------------------
        for (x_row, normed_row) in scratch
            .x
            .chunks_exact(dims.hidden)
            .zip(scratch.normed.chunks_exact_mut(dims.hidden))
        {
            normed_row.copy_from_slice(x_row);
            rms_norm_inplace(normed_row, &weights.post_norm, dims.eps);
        }
        scratch
            .prepared
            .prepare(
                &scratch.normed,
                rows,
                dims.hidden,
                meta.mlp_q8,
                meta.mlp_q8k,
            )
            .map_err(|error| context(&error))?;
        scratch
            .prepared
            .matmul_group(
                &scratch.normed,
                [
                    (&weights.gate, &mut scratch.gate[..rows * dims.ffn]),
                    (&weights.up, &mut scratch.up[..rows * dims.ffn]),
                ],
                pool,
            )
            .map_err(|error| context(&error))?;
        silu_inplace(&mut scratch.gate[..rows * dims.ffn]);
        vec_mul_inplace(
            &scratch.up[..rows * dims.ffn],
            &mut scratch.gate[..rows * dims.ffn],
        );
        scratch
            .prepared
            .prepare(&scratch.gate, rows, dims.ffn, meta.down_q8, meta.down_q8k)
            .map_err(|error| context(&error))?;
        scratch
            .prepared
            .matmul(
                &weights.down,
                &scratch.gate,
                &mut scratch.proj[..rows * dims.hidden],
                pool,
            )
            .map_err(|error| context(&error))?;
        for (projected, x_row) in scratch
            .proj
            .chunks_exact(dims.hidden)
            .zip(scratch.x.chunks_exact_mut(dims.hidden))
        {
            vec_add_into(projected, x_row);
        }
        Ok(())
    }
}

// =============================================================================
// M-RoPE
// =============================================================================

/// Build the half-split cos/sin table of one segment at `pos`.
///
/// `scale` is `theta^(-2/seg_dim)`, so the running product reproduces
/// `inv_freq = [theta^(-2i/seg_dim)]` without a `powf` per entry.
fn build_rope_table(pos: usize, half: usize, scale: f32, cos: &mut [f32], sin: &mut [f32]) {
    let mut angle = pos as f32;
    for index in 0..half {
        let (cosine, sine) = rope_sin_cos(angle);
        cos[index] = cosine;
        sin[index] = sine;
        angle *= scale;
    }
}

/// Apply the three-segment M-RoPE to `x` laid out as `[rows, heads, head_dim]`.
///
/// One segment is exactly `rope_neox(head_dim = seg_dim, theta = seg_theta)` at
/// that segment's own position, so the 128-dim head is never rotated as a
/// single block: the cos/sin table is built per `(row, segment)` with
/// [`rope_sin_cos`] and the shared rotation kernel is applied to each head's
/// segment. (`rope_neox_inplace_with_table` is *not* usable here — it is the
/// Breeze variant that round-trips the products through BF16.)
#[allow(clippy::too_many_arguments)]
fn rope_segments_inplace(
    x: &mut [f32],
    rows: usize,
    heads: usize,
    dims: Dims,
    positions: &[[usize; 3]],
    cos: &mut [f32],
    sin: &mut [f32],
) {
    debug_assert_eq!(positions.len(), rows, "one M-RoPE index per row");
    let head_dim = dims.head_dim;
    let segments = dims.rope_segments();
    for (row, position) in positions.iter().enumerate().take(rows) {
        let row_base = row * heads * head_dim;
        for (segment, &(offset, width, scale)) in segments.iter().enumerate() {
            let half = width / 2;
            build_rope_table(
                position[segment],
                half,
                scale,
                &mut cos[..half],
                &mut sin[..half],
            );
            for head in 0..heads {
                let start = row_base + head * head_dim + offset;
                rope_neox_inplace_scalar(
                    &mut x[start..start + width],
                    1,
                    width,
                    &cos[..half],
                    &sin[..half],
                );
            }
        }
    }
}

/// RMSNorm the two 64-wide halves of every head, then rotate.
///
/// Upstream normalizes `q.chunk(2, -1)` with `q_norm` (t half) and `q_norm_hw`
/// (hw half) while the head is still a single 128-wide vector, and only then
/// splits the hw half into h and w. That order matters: `q_norm_hw` sees 64
/// values, not two independent 32-value groups.
#[allow(clippy::too_many_arguments)]
fn head_norm_and_rope(
    x: &mut [f32],
    rows: usize,
    heads: usize,
    dims: Dims,
    norm_t: &[f32],
    norm_hw: &[f32],
    positions: &[[usize; 3]],
    cos: &mut [f32],
    sin: &mut [f32],
) {
    let head_dim = dims.head_dim;
    let rope_t = dims.rope_t();
    for row in 0..rows {
        let row_base = row * heads * head_dim;
        for head in 0..heads {
            let start = row_base + head * head_dim;
            rms_norm_inplace(&mut x[start..start + rope_t], norm_t, dims.eps);
            rms_norm_inplace(&mut x[start + rope_t..start + head_dim], norm_hw, dims.eps);
        }
    }
    rope_segments_inplace(x, rows, heads, dims, positions, cos, sin);
}

// =============================================================================
// Attention
// =============================================================================

/// Scaled dot-product attention over `[prefix keys] ++ [current block]`.
///
/// `q` is `[rows, q_dim]`, `k_cur`/`v_cur` are `[rows, kv_dim]` (token-major,
/// the layout the projections produce) and `out` is `[rows, q_dim]`. The
/// prefix, when present, is head-major `[kv_heads, len, head_dim]`, which is
/// what [`PrefixCache`] stores.
///
/// The block mask is causal for the understanding branch and fully
/// bidirectional for the generation branch (contract §3.2).
#[allow(clippy::too_many_arguments)]
fn attend(
    q: &[f32],
    k_cur: &[f32],
    v_cur: &[f32],
    prefix: Option<(&[f32], &[f32], usize)>,
    out: &mut [f32],
    rows: usize,
    causal: bool,
    dims: Dims,
    pool: &ComputePool,
) {
    let head_dim = dims.head_dim;
    let q_dim = dims.q_dim();
    let kv_dim = dims.kv_dim();
    let query_heads = dims.query_heads;
    let group = dims.group();
    let scale = 1.0 / (head_dim as f32).sqrt();
    let (prefix_k, prefix_v, prefix_len) = match prefix {
        Some((k, v, len)) => (k, v, len),
        None => (&[][..], &[][..], 0),
    };
    let total = rows * query_heads;
    let out_ptr = out.as_mut_ptr();

    pool.compute(|ith, nth| {
        // One scores buffer per worker: `rows * query_heads` tasks would
        // otherwise allocate a few thousand times per layer.
        let mut scores = vec![0.0f32; prefix_len + rows];
        let mut block = vec![0.0f32; head_dim];
        let mut task = ith;
        while task < total {
            let row = task / query_heads;
            let head = task % query_heads;
            let kv_head = head / group;
            let q_offset = row * q_dim + head * head_dim;
            let q_vec = &q[q_offset..q_offset + head_dim];

            for (index, slot) in scores[..prefix_len].iter_mut().enumerate() {
                let k_offset = (kv_head * prefix_len + index) * head_dim;
                *slot = dot_f32(q_vec, &prefix_k[k_offset..k_offset + head_dim], head_dim) * scale;
            }
            let visible = if causal { row + 1 } else { rows };
            for index in 0..visible {
                let k_offset = index * kv_dim + kv_head * head_dim;
                scores[prefix_len + index] =
                    dot_f32(q_vec, &k_cur[k_offset..k_offset + head_dim], head_dim) * scale;
            }
            let n_keys = prefix_len + visible;
            softmax_inplace(&mut scores[..n_keys]);

            // SAFETY: every task index is handled by exactly one worker
            // (`task` starts at `ith` and steps by `nth`), and a task owns the
            // `head_dim` floats of one (row, head) output slot, so the derived
            // slices are disjoint. Same audited pattern as
            // `Weight::quantize_and_matmul_with_scratch`.
            let output =
                unsafe { std::slice::from_raw_parts_mut(out_ptr.add(task * head_dim), head_dim) };
            if prefix_len == 0 {
                attention_value_reduce(
                    v_cur,
                    &scores[..visible],
                    output,
                    0,
                    kv_dim,
                    kv_head * head_dim,
                    visible,
                    head_dim,
                );
            } else {
                attention_value_reduce(
                    prefix_v,
                    &scores[..prefix_len],
                    output,
                    kv_head * prefix_len * head_dim,
                    head_dim,
                    0,
                    prefix_len,
                    head_dim,
                );
                attention_value_reduce(
                    v_cur,
                    &scores[prefix_len..n_keys],
                    &mut block,
                    0,
                    kv_dim,
                    kv_head * head_dim,
                    visible,
                    head_dim,
                );
                vec_add_into(&block, output);
            }
            task += nth;
        }
    });
}

// =============================================================================
// Scratch
// =============================================================================

/// Buffers reused across the layers of one forward call.
struct Scratch {
    x: Vec<f32>,
    normed: Vec<f32>,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attn: Vec<f32>,
    proj: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    cos: Vec<f32>,
    sin: Vec<f32>,
    prepared: PreparedRows,
}

impl Scratch {
    fn new(dims: Dims, rows: usize) -> Self {
        let widest = dims.hidden.max(dims.ffn);
        Self {
            x: vec![0.0; rows * dims.hidden],
            normed: vec![0.0; rows * dims.hidden],
            q: vec![0.0; rows * dims.q_dim()],
            k: vec![0.0; rows * dims.kv_dim()],
            v: vec![0.0; rows * dims.kv_dim()],
            attn: vec![0.0; rows * dims.q_dim()],
            proj: vec![0.0; rows * dims.hidden],
            gate: vec![0.0; rows * dims.ffn],
            up: vec![0.0; rows * dims.ffn],
            cos: vec![0.0; dims.rope_t()],
            sin: vec![0.0; dims.rope_t()],
            prepared: PreparedRows::new(rows, widest),
        }
    }
}

/// Transpose the token-major `[rows, kv_dim]` projections of one layer into the
/// head-major `[kv_heads, rows, head_dim]` layout [`PrefixCache`] promises.
fn store_kv(k: &[f32], v: &[f32], rows: usize, dims: Dims) -> LayerKv {
    let head_dim = dims.head_dim;
    let kv_dim = dims.kv_dim();
    let span = dims.kv_heads * rows * head_dim;
    let mut cache = LayerKv {
        k: vec![0.0; span],
        v: vec![0.0; span],
    };
    for row in 0..rows {
        for head in 0..dims.kv_heads {
            let source = row * kv_dim + head * head_dim;
            let target = (head * rows + row) * head_dim;
            cache.k[target..target + head_dim].copy_from_slice(&k[source..source + head_dim]);
            cache.v[target..target + head_dim].copy_from_slice(&v[source..source + head_dim]);
        }
    }
    cache
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::tensor::{MetaValue, TensorInfo};
    use std::collections::HashMap;

    /// Tiny stand-in for the real geometry: one layer, two query heads, one KV
    /// head, 8-wide heads (t = 4, h = w = 2), so `q_dim == hidden` the way the
    /// 8B model has `32 * 128 == 4096`. Exercises every code path the real
    /// geometry takes, on a few kilobytes of synthetic weights.
    fn tiny_dims() -> Dims {
        Dims::build(16, 1, 2, 1, 8, 32, 16)
    }

    /// Deterministic pseudo-random values so the tests never depend on a seed
    /// from `rand` and stay reproducible across platforms.
    fn pseudo_random(seed: &mut u64, count: usize, scale: f32) -> Vec<f32> {
        (0..count)
            .map(|_| {
                *seed = seed
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let unit = ((*seed >> 33) as f32) / ((u32::MAX >> 1) as f32);
                (unit * 2.0 - 1.0) * scale
            })
            .collect()
    }

    #[derive(Default)]
    struct MapSource {
        info: HashMap<String, TensorInfo>,
        data: HashMap<String, Vec<u8>>,
        /// Decoded F32 payloads, so a test can drive a naive reference over the
        /// same weights the trunk loads.
        values: HashMap<String, Vec<f32>>,
    }

    impl MapSource {
        fn add(&mut self, name: &str, dims: &[u64], values: &[f32]) {
            self.info.insert(
                name.to_string(),
                TensorInfo {
                    name: name.to_string(),
                    dims: dims.to_vec(),
                    ggml_type: GGMLType::F32,
                    offset: 0,
                },
            );
            self.data.insert(
                name.to_string(),
                values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect(),
            );
            self.values.insert(name.to_string(), values.to_vec());
        }

        fn add_ones(&mut self, name: &str, dims: &[u64]) {
            let count: u64 = dims.iter().product();
            self.add(name, dims, &vec![1.0; count as usize]);
        }

        /// Add a tensor in a non-F32 GGUF format. Only Q8_0 is fabricated; the
        /// other supported formats share its kernel/prepare plumbing
        /// (`Q4_0` via the Q8_0 activation, `Q6_K` via Q8_K) or need no
        /// activation quantization at all (`F16`).
        fn add_typed(&mut self, name: &str, dims: &[u64], values: &[f32], ggml_type: GGMLType) {
            assert_eq!(ggml_type, GGMLType::Q8_0, "only Q8_0 is fabricated");
            self.info.insert(
                name.to_string(),
                TensorInfo {
                    name: name.to_string(),
                    dims: dims.to_vec(),
                    ggml_type,
                    offset: 0,
                },
            );
            self.data
                .insert(name.to_string(), quantize_q8_0_weight(values));
            self.values.insert(name.to_string(), values.to_vec());
        }
    }

    /// GGUF Q8_0 blocks: `{ d: f16, qs: [i8; 32] }`, `q = round(x / d)`.
    fn quantize_q8_0_weight(values: &[f32]) -> Vec<u8> {
        assert_eq!(values.len() % 32, 0, "Q8_0 rows are whole 32-blocks");
        let mut bytes = Vec::with_capacity(values.len() + values.len() / 16);
        for start in (0..values.len()).step_by(32) {
            let block = &values[start..start + 32];
            let max = block.iter().fold(0.0f32, |max, value| max.max(value.abs()));
            let d = max / 127.0;
            let inverse = if d == 0.0 { 0.0 } else { 1.0 / d };
            bytes.extend_from_slice(&crate::ops::f32_to_f16(d).to_le_bytes());
            for value in block {
                let quantized = (value * inverse).round().clamp(-127.0, 127.0) as i8;
                bytes.push(quantized as u8);
            }
        }
        bytes
    }

    impl TensorSource for MapSource {
        fn metadata(&self, _: &str) -> Option<&MetaValue> {
            None
        }
        fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
            self.info.get(name)
        }
        fn tensor_slice(&self, name: &str) -> Option<&[u8]> {
            self.data.get(name).map(|bytes| bytes.as_slice())
        }
    }

    /// Fabricate a complete trunk's worth of tensors for `dims`, all F32.
    fn synthetic_source(dims: Dims) -> MapSource {
        synthetic_source_with(dims, GGMLType::F32)
    }

    /// Same, but with every linear stored in `linear`.
    fn synthetic_source_with(dims: Dims, linear: GGMLType) -> MapSource {
        let mut source = MapSource::default();
        let mut seed = 0x5eed_1234_u64;
        let hidden = dims.hidden as u64;
        let q_dim = dims.q_dim() as u64;
        let kv_dim = dims.kv_dim() as u64;
        let ffn = dims.ffn as u64;
        let head_norm = dims.rope_t() as u64;

        source.add(
            "language_model.model.embed_tokens.weight",
            &[hidden, dims.vocab as u64],
            &pseudo_random(&mut seed, dims.hidden * dims.vocab, 0.1),
        );
        for name in [
            "language_model.model.norm.weight",
            "language_model.model.norm_mot_gen.weight",
        ] {
            source.add_ones(name, &[hidden]);
        }

        for layer in 0..dims.layers {
            for branch in [Branch::Und, Branch::Gen] {
                let names = LayerNames::new(layer, branch);
                source.add_ones(&names.input_norm, &[hidden]);
                source.add_ones(&names.post_norm, &[hidden]);
                for name in [
                    &names.q_norm,
                    &names.q_norm_hw,
                    &names.k_norm,
                    &names.k_norm_hw,
                ] {
                    source.add_ones(name, &[head_norm]);
                }
                for (name, out) in [
                    (&names.q_proj, q_dim),
                    (&names.k_proj, kv_dim),
                    (&names.v_proj, kv_dim),
                    (&names.o_proj, hidden),
                    (&names.gate, ffn),
                    (&names.up, ffn),
                    (&names.down, hidden),
                ] {
                    let n_in = if name == &names.down { ffn } else { hidden };
                    let count = (n_in * out) as usize;
                    let values = pseudo_random(&mut seed, count, 0.05);
                    match linear {
                        GGMLType::F32 => source.add(name, &[n_in, out], &values),
                        other => source.add_typed(name, &[n_in, out], &values, other),
                    }
                }
            }
        }
        source
    }

    /// Straight transcription of upstream's `apply_rotary_pos_emb` per segment:
    /// `emb = cat(freqs, freqs)`, `rotate_half`, `angle_i = pos * theta^(-2i/d)`.
    fn reference_rope(
        x: &[f32],
        rows: usize,
        heads: usize,
        dims: Dims,
        positions: &[[usize; 3]],
    ) -> Vec<f32> {
        let head_dim = dims.head_dim;
        let thetas = [ROPE_THETA_T, ROPE_THETA_HW, ROPE_THETA_HW];
        let segments = dims.rope_segments();
        let mut out = x.to_vec();
        for (row, position) in positions.iter().enumerate().take(rows) {
            for head in 0..heads {
                let base = (row * heads + head) * head_dim;
                for (segment, &(offset, width, _)) in segments.iter().enumerate() {
                    let half = width / 2;
                    let pos = position[segment] as f32;
                    for i in 0..half {
                        let inv_freq = thetas[segment].powf(-2.0 * i as f32 / width as f32);
                        let angle = pos * inv_freq;
                        let (sin, cos) = (angle.sin(), angle.cos());
                        let lo = x[base + offset + i];
                        let hi = x[base + offset + i + half];
                        out[base + offset + i] = lo * cos - hi * sin;
                        out[base + offset + i + half] = lo * sin + hi * cos;
                    }
                }
            }
        }
        out
    }

    /// Element-wise comparison with a relative tolerance: the NEON/aarch64
    /// rotation kernels fuse `mul + add`, so a 1-ULP difference against a
    /// two-rounding scalar reference is expected.
    fn assert_close(got: &[f32], want: &[f32], tolerance: f32, what: &str) {
        assert_eq!(got.len(), want.len(), "{what}: length mismatch");
        for (index, (got, want)) in got.iter().zip(want).enumerate() {
            let allowed = tolerance * (1.0 + want.abs());
            assert!(
                (got - want).abs() <= allowed,
                "{what}: element {index}: got {got}, want {want}"
            );
        }
    }

    #[test]
    fn rope_three_segments_match_half_split_reference() {
        for dims in [tiny_dims(), Dims::contract()] {
            let heads = 2;
            let rows = 3;
            let positions = [[0usize, 0, 0], [1, 2, 3], [5, 1, 4]];
            let mut seed = 7_u64;
            let mut x = pseudo_random(&mut seed, rows * heads * dims.head_dim, 1.0);
            let want = reference_rope(&x, rows, heads, dims, &positions);

            let mut cos = vec![0.0f32; dims.rope_t()];
            let mut sin = vec![0.0f32; dims.rope_t()];
            rope_segments_inplace(&mut x, rows, heads, dims, &positions, &mut cos, &mut sin);

            assert_close(
                &x,
                &want,
                2e-5,
                &format!("three-segment rope, head_dim {}", dims.head_dim),
            );
        }
    }

    #[test]
    fn rope_positions_are_per_segment() {
        // pos = 0 leaves every segment untouched, and advancing only the t
        // position must not touch the h/w slices: rotating the whole 128-dim
        // head as one block would move all of them.
        let dims = tiny_dims();
        let mut seed = 11_u64;
        let original = pseudo_random(&mut seed, dims.head_dim, 1.0);
        let mut cos = vec![0.0f32; dims.rope_t()];
        let mut sin = vec![0.0f32; dims.rope_t()];

        let mut rotated = original.clone();
        rope_segments_inplace(&mut rotated, 1, 1, dims, &[[0, 0, 0]], &mut cos, &mut sin);
        assert_eq!(rotated, original, "all-zero positions must be the identity");

        let mut rotated = original.clone();
        rope_segments_inplace(&mut rotated, 1, 1, dims, &[[7, 0, 0]], &mut cos, &mut sin);
        let want = reference_rope(&original, 1, 1, dims, &[[7, 0, 0]]);
        let rope_t = dims.rope_t();
        assert_close(
            &rotated[..rope_t],
            &want[..rope_t],
            2e-5,
            "t slice must rotate with pos = t",
        );
        assert_close(
            &rotated[rope_t..],
            &original[rope_t..],
            2e-5,
            "h/w slices must stay put when only t advances",
        );
    }

    #[test]
    fn tiny_trunk_prefix_produces_finite_hidden_and_head_major_kv() {
        let dims = tiny_dims();
        let source = synthetic_source(dims);
        let trunk = Trunk::build(&source, dims).expect("synthetic trunk must validate");
        let pool = ComputePool::new(2);
        let tokens = [3u32, 1, 4];
        let (cache, hidden) = trunk
            .encode_prefix(&source, &tokens, &pool)
            .expect("prefix forward must succeed");

        assert_eq!(cache.len, tokens.len());
        assert_eq!(cache.layers.len(), dims.layers);
        let span = dims.kv_heads * tokens.len() * dims.head_dim;
        for (layer, kv) in cache.layers.iter().enumerate() {
            assert_eq!(kv.k.len(), span, "layer {layer} k shape");
            assert_eq!(kv.v.len(), span, "layer {layer} v shape");
            assert!(
                kv.k.iter()
                    .chain(kv.v.iter())
                    .all(|value| value.is_finite()),
                "layer {layer} KV must be finite"
            );
        }
        assert_eq!(hidden.len(), tokens.len() * dims.hidden);
        assert!(
            hidden.iter().all(|value| value.is_finite()),
            "prefix hidden state must be finite"
        );
        assert!(
            hidden.iter().any(|value| *value != 0.0),
            "prefix hidden state must not be all zeros"
        );
    }

    #[test]
    fn denoise_leaves_the_prefix_cache_untouched() {
        let dims = tiny_dims();
        let source = synthetic_source(dims);
        let trunk = Trunk::build(&source, dims).expect("synthetic trunk must validate");
        let pool = ComputePool::new(2);
        let tokens = [3u32, 1, 4];
        let (cache, _) = trunk
            .encode_prefix(&source, &tokens, &pool)
            .expect("prefix forward must succeed");

        let before: Vec<(Vec<f32>, Vec<f32>)> = cache
            .layers
            .iter()
            .map(|kv| (kv.k.clone(), kv.v.clone()))
            .collect();

        let token_w = 2;
        let rows = 4;
        let mut seed = 99_u64;
        let embeds = pseudo_random(&mut seed, rows * dims.hidden, 0.1);
        let out = trunk
            .denoise(&source, &embeds, tokens.len(), token_w, &cache, &pool)
            .expect("denoise must succeed");

        assert_eq!(out.len(), rows * dims.hidden);
        assert!(
            out.iter().all(|value| value.is_finite()),
            "denoise output must be finite"
        );
        assert!(
            out.iter().any(|value| *value != 0.0),
            "denoise output must not be all zeros"
        );
        for (layer, kv) in cache.layers.iter().enumerate() {
            assert_eq!(kv.k, before[layer].0, "layer {layer} prefix K was mutated");
            assert_eq!(kv.v, before[layer].1, "layer {layer} prefix V was mutated");
        }
    }

    #[test]
    fn denoise_rejects_a_mismatched_prefix_length() {
        let dims = tiny_dims();
        let source = synthetic_source(dims);
        let trunk = Trunk::build(&source, dims).expect("synthetic trunk must validate");
        let pool = ComputePool::new(2);
        let (cache, _) = trunk
            .encode_prefix(&source, &[1u32, 2], &pool)
            .expect("prefix forward must succeed");
        let error = trunk
            .denoise(&source, &[0.0; 16], 5, 1, &cache, &pool)
            .expect_err("a text_len that disagrees with the cache must be rejected");
        assert!(error.contains("prefix cache"), "{error}");
    }

    // -----------------------------------------------------------------
    // Naive transcription of contract §3, used to cross-check the
    // optimized path. Plain loops, f64 accumulation, no scratch reuse.
    // -----------------------------------------------------------------

    fn naive_rms(values: &mut [f32], weight: &[f32], eps: f32) {
        let sum_sq: f64 = values.iter().map(|value| f64::from(*value).powi(2)).sum();
        let scale = 1.0 / ((sum_sq / values.len() as f64) as f32 + eps).sqrt();
        for (value, weight) in values.iter_mut().zip(weight) {
            *value = *value * scale * *weight;
        }
    }

    fn naive_dot(a: &[f32], b: &[f32]) -> f32 {
        let acc: f64 = a
            .iter()
            .zip(b)
            .map(|(a, b)| f64::from(*a) * f64::from(*b))
            .sum();
        acc as f32
    }

    /// `[rows, n_in] -> [rows, n_out]` for a GGUF F32 weight stored as
    /// `[n_out, n_in]` row-major.
    fn naive_linear(
        values: &HashMap<String, Vec<f32>>,
        name: &str,
        n_in: usize,
        n_out: usize,
        input: &[f32],
    ) -> Vec<f32> {
        let weight = &values[name];
        assert_eq!(weight.len(), n_in * n_out, "{name} shape");
        (0..n_out)
            .map(|row| naive_dot(&weight[row * n_in..(row + 1) * n_in], input))
            .collect()
    }

    fn naive_softmax(scores: &mut [f32]) {
        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f64;
        for score in scores.iter_mut() {
            *score = (*score - max).exp();
            sum += f64::from(*score);
        }
        for score in scores.iter_mut() {
            *score = (*score as f64 / sum) as f32;
        }
    }

    /// One decoder layer of the contract, in the same token-major layout the
    /// projections produce. Returns the head-major K/V it would cache.
    #[allow(clippy::too_many_arguments)]
    fn naive_layer(
        x: &mut [f32],
        values: &HashMap<String, Vec<f32>>,
        layer: usize,
        branch: Branch,
        dims: Dims,
        positions: &[[usize; 3]],
        prefix: Option<(&[f32], &[f32], usize)>,
        causal: bool,
    ) -> (Vec<f32>, Vec<f32>) {
        let names = LayerNames::new(layer, branch);
        let rows = positions.len();
        let hidden = dims.hidden;
        let q_dim = dims.q_dim();
        let kv_dim = dims.kv_dim();
        let head_dim = dims.head_dim;
        let rope_t = dims.rope_t();
        let group = dims.group();

        // Attention projections on the input-layernorm output.
        let mut q = vec![0.0f32; rows * q_dim];
        let mut k = vec![0.0f32; rows * kv_dim];
        let mut v = vec![0.0f32; rows * kv_dim];
        for row in 0..rows {
            let mut normed = x[row * hidden..(row + 1) * hidden].to_vec();
            naive_rms(&mut normed, &values[&names.input_norm], dims.eps);
            q[row * q_dim..(row + 1) * q_dim].copy_from_slice(&naive_linear(
                values,
                &names.q_proj,
                hidden,
                q_dim,
                &normed,
            ));
            k[row * kv_dim..(row + 1) * kv_dim].copy_from_slice(&naive_linear(
                values,
                &names.k_proj,
                hidden,
                kv_dim,
                &normed,
            ));
            v[row * kv_dim..(row + 1) * kv_dim].copy_from_slice(&naive_linear(
                values,
                &names.v_proj,
                hidden,
                kv_dim,
                &normed,
            ));
        }

        // Per-head RMSNorm on the t half and the hw half, before the h/w split.
        for row in 0..rows {
            for head in 0..dims.query_heads {
                let base = row * q_dim + head * head_dim;
                naive_rms(
                    &mut q[base..base + rope_t],
                    &values[&names.q_norm],
                    dims.eps,
                );
                naive_rms(
                    &mut q[base + rope_t..base + head_dim],
                    &values[&names.q_norm_hw],
                    dims.eps,
                );
            }
            for head in 0..dims.kv_heads {
                let base = row * kv_dim + head * head_dim;
                naive_rms(
                    &mut k[base..base + rope_t],
                    &values[&names.k_norm],
                    dims.eps,
                );
                naive_rms(
                    &mut k[base + rope_t..base + head_dim],
                    &values[&names.k_norm_hw],
                    dims.eps,
                );
            }
        }

        let q = reference_rope(&q, rows, dims.query_heads, dims, positions);
        let k = reference_rope(&k, rows, dims.kv_heads, dims, positions);

        // Head-major KV, the layout `PrefixCache` promises.
        let span = dims.kv_heads * rows * head_dim;
        let mut cached_k = vec![0.0f32; span];
        let mut cached_v = vec![0.0f32; span];
        for row in 0..rows {
            for head in 0..dims.kv_heads {
                let source = row * kv_dim + head * head_dim;
                let target = (head * rows + row) * head_dim;
                cached_k[target..target + head_dim].copy_from_slice(&k[source..source + head_dim]);
                cached_v[target..target + head_dim].copy_from_slice(&v[source..source + head_dim]);
            }
        }

        // Attention.
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut attn = vec![0.0f32; rows * q_dim];
        for row in 0..rows {
            let visible = if causal { row + 1 } else { rows };
            for head in 0..dims.query_heads {
                let kv_head = head / group;
                let q_vec = &q[row * q_dim + head * head_dim..][..head_dim];
                let mut scores = Vec::with_capacity(rows + prefix.map_or(0, |p| p.2));
                let mut keys: Vec<&[f32]> = Vec::new();
                if let Some((prefix_k, prefix_v, prefix_len)) = prefix {
                    for index in 0..prefix_len {
                        let at = (kv_head * prefix_len + index) * head_dim;
                        scores.push(naive_dot(q_vec, &prefix_k[at..at + head_dim]) * scale);
                        keys.push(&prefix_v[at..at + head_dim]);
                    }
                }
                for index in 0..visible {
                    let at = index * kv_dim + kv_head * head_dim;
                    scores.push(naive_dot(q_vec, &k[at..at + head_dim]) * scale);
                    keys.push(&v[at..at + head_dim]);
                }
                naive_softmax(&mut scores);
                let out = &mut attn[row * q_dim + head * head_dim..][..head_dim];
                for (score, key) in scores.iter().zip(&keys) {
                    for (slot, value) in out.iter_mut().zip(key.iter()) {
                        *slot += score * value;
                    }
                }
            }
        }

        // o_proj + residual.
        for row in 0..rows {
            let projected = naive_linear(
                values,
                &names.o_proj,
                q_dim,
                hidden,
                &attn[row * q_dim..(row + 1) * q_dim],
            );
            for (slot, value) in x[row * hidden..(row + 1) * hidden]
                .iter_mut()
                .zip(&projected)
            {
                *slot += value;
            }
        }

        // SwiGLU FFN + residual.
        for row in 0..rows {
            let mut normed = x[row * hidden..(row + 1) * hidden].to_vec();
            naive_rms(&mut normed, &values[&names.post_norm], dims.eps);
            let gate = naive_linear(values, &names.gate, hidden, dims.ffn, &normed);
            let up = naive_linear(values, &names.up, hidden, dims.ffn, &normed);
            let activated: Vec<f32> = gate
                .iter()
                .zip(&up)
                .map(|(gate, up)| (gate / (1.0 + (-gate).exp())) * up)
                .collect();
            let projected = naive_linear(values, &names.down, dims.ffn, hidden, &activated);
            for (slot, value) in x[row * hidden..(row + 1) * hidden]
                .iter_mut()
                .zip(&projected)
            {
                *slot += value;
            }
        }
        (cached_k, cached_v)
    }

    #[test]
    fn tiny_trunk_matches_a_naive_reference() {
        let dims = tiny_dims();
        let source = synthetic_source(dims);
        let trunk = Trunk::build(&source, dims).expect("synthetic trunk must validate");
        let pool = ComputePool::new(2);
        let tokens = [3u32, 1, 4];
        let values = &source.values;

        // ---- understanding branch (causal, no prefix) -------------------
        let (cache, hidden) = trunk
            .encode_prefix(&source, &tokens, &pool)
            .expect("prefix forward must succeed");
        let embed = &values["language_model.model.embed_tokens.weight"];
        let mut want = Vec::new();
        for &token in &tokens {
            let at = token as usize * dims.hidden;
            want.extend_from_slice(&embed[at..at + dims.hidden]);
        }
        let positions: Vec<[usize; 3]> = (0..tokens.len()).map(|t| [t, 0, 0]).collect();
        let (want_k, want_v) = naive_layer(
            &mut want,
            values,
            0,
            Branch::Und,
            dims,
            &positions,
            None,
            true,
        );
        for row in want.chunks_exact_mut(dims.hidden) {
            naive_rms(row, &values["language_model.model.norm.weight"], dims.eps);
        }
        assert_close(&hidden, &want, 2e-4, "prefix hidden state");
        assert_close(&cache.layers[0].k, &want_k, 2e-4, "prefix K cache");
        assert_close(&cache.layers[0].v, &want_v, 2e-4, "prefix V cache");

        // ---- generation branch (bidirectional over prefix + block) ------
        let token_w = 2;
        let rows = 4;
        let mut seed = 1234_u64;
        let embeds = pseudo_random(&mut seed, rows * dims.hidden, 0.1);
        let got = trunk
            .denoise(&source, &embeds, tokens.len(), token_w, &cache, &pool)
            .expect("denoise must succeed");

        let image_positions: Vec<[usize; 3]> = (0..rows)
            .map(|index| [tokens.len(), index / token_w, index % token_w])
            .collect();
        let mut want = embeds.clone();
        naive_layer(
            &mut want,
            values,
            0,
            Branch::Gen,
            dims,
            &image_positions,
            Some((&cache.layers[0].k, &cache.layers[0].v, tokens.len())),
            false,
        );
        for row in want.chunks_exact_mut(dims.hidden) {
            naive_rms(
                row,
                &values["language_model.model.norm_mot_gen.weight"],
                dims.eps,
            );
        }
        assert_close(&got, &want, 2e-4, "denoise hidden state");
    }

    #[test]
    fn q8_0_linears_run_the_quantized_kernel() {
        // `hidden` / `ffn` are multiples of 32 so the Q8_0 block layout is
        // valid; the real model uses 4096 / 12288, which are multiples of 256.
        let dims = Dims::build(32, 1, 2, 1, 16, 64, 8);
        let f32_source = synthetic_source(dims);
        let q8_source = synthetic_source_with(dims, GGMLType::Q8_0);
        let f32_trunk = Trunk::build(&f32_source, dims).expect("F32 trunk must validate");
        let q8_trunk = Trunk::build(&q8_source, dims).expect("Q8_0 trunk must validate");
        let pool = ComputePool::new(2);
        let tokens = [1u32, 2, 3];
        let (_, reference) = f32_trunk
            .encode_prefix(&f32_source, &tokens, &pool)
            .expect("F32 prefix forward must succeed");
        let (cache, quantized) = q8_trunk
            .encode_prefix(&q8_source, &tokens, &pool)
            .expect("Q8_0 prefix forward must succeed");

        assert!(quantized.iter().all(|value| value.is_finite()));
        assert_eq!(
            cache.layers[0].k.len(),
            dims.kv_heads * tokens.len() * dims.head_dim
        );
        let scale = (reference.iter().map(|value| value * value).sum::<f32>()
            / reference.len() as f32)
            .sqrt();
        let max_diff = reference
            .iter()
            .zip(&quantized)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff > 0.0,
            "Q8_0 must not reproduce the F32 result bit for bit"
        );
        assert!(
            max_diff < 0.1 * scale,
            "Q8_0 drift {max_diff} must stay well under the output scale {scale}"
        );
    }

    #[test]
    fn build_rejects_a_truncated_contract() {
        let dims = tiny_dims();
        let mut source = synthetic_source(dims);
        source
            .info
            .remove("language_model.model.layers.0.self_attn.q_proj_mot_gen.weight");
        let error = Trunk::build(&source, dims)
            .err()
            .expect("a missing generation-branch projection must be rejected");
        assert!(error.contains("q_proj_mot_gen"), "{error}");
    }
}
