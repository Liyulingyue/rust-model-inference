use super::super::contract::{
    require_gemma4_token_table, require_string, require_tensor, require_tensor_any,
};
use crate::core::tensor::{GGMLType, MetaValue, MetaValueType, TensorSource};

/// Architecture constants shared across all Gemma 4 variants. Values
/// that are uniform across the family (vocab size, RoPE base, eps) live
/// here; everything else is read from GGUF metadata at load time and
/// stored on `Gemma4Config`.
pub(super) const VOCAB: usize = 262_144;
pub(super) const EPS: f32 = 1e-6;

/// Canonical Gemma 4 context length for the E2B/E4B variants, kept as
/// a compile-time default for tests and the historical 131072 boundary
/// checks. The 12B variant advertises 262144 in GGUF; that value is
/// stored on `Gemma4Config::n_ctx` at load time.
pub(super) const CONTEXT: usize = 131_072;

/// Sanity upper bound for `gemma4.context_length`. Real Gemma 4 exports
/// sit at 131072 (E2B/E4B) or 262144 (12B). Anything beyond this is
/// almost certainly a corrupted metadata read rather than a real model.
pub(super) const MAX_CONTEXT: usize = 1 << 24;

/// E2B/E4B per-layer projection width. Gemma 4 12B disables the
/// per-layer projection entirely (`embedding_length_per_layer_input = 0`).
/// Kept as a constant for E2B/E4B tests and a default when the metadata
/// is missing.
pub(super) const PER_LAYER: usize = 256;

#[derive(Debug, Clone, PartialEq)]
pub struct Gemma4Config {
    pub layers: usize,
    pub embd: usize,
    /// Total query heads per layer. E2B/E4B = 8, 12B = 16.
    pub n_heads: usize,
    /// Per-layer KV head count. Length == `layers`. E2B uses uniform
    /// `vec![1; layers]`; 12B alternates `[8,8,8,8,8,1,8,8,...]`.
    pub kv_heads_per_layer: Vec<usize>,
    pub vocab: usize,
    pub full_head_dim: usize,
    pub swa_head_dim: usize,
    pub shared_kv_layers: usize,
    pub sliding_window: usize,
    pub logit_softcap: f32,
    /// Maximum supported context length, taken from the GGUF
    /// `gemma4.context_length` metadata (131072 for E2B/E4B, 262144 for
    /// 12B). Stored on the config so the runtime bound check, scratch
    /// sizing and prefill cap all read the model's own value rather
    /// than a compile-time constant.
    pub n_ctx: usize,
    /// Per-layer FFN width. Length == `layers`. E2B is heterogeneous
    /// ([6144; 15] ++ [12288; 20]); 12B is uniform (15360 × 48).
    pub ffn_per_layer: Vec<usize>,
    /// Per-layer sliding-window boolean. `true` = SWA, `false` = full
    /// attention. Length == `layers`.
    pub swa_pattern: Vec<bool>,
    /// Per-layer token-embedding width for the optional `per_layer_*`
    /// projection. E2B/E4B = 256 (projection enabled); 12B = 0
    /// (projection disabled, no `per_layer_*` tensors stored).
    pub per_layer_width: usize,
    /// RoPE base for full-attention layers.
    pub rope_freq_base: f32,
    /// RoPE base for SWA layers.
    pub rope_freq_base_swa: f32,
}

impl Gemma4Config {
    /// `layers - shared_kv_layers`: count of leading layers with their own
    /// (non-shared) KV cache.
    pub fn base_kv_layers(&self) -> usize {
        self.layers - self.shared_kv_layers
    }

    /// Max FFN width across all layers (drives scratch buffer sizing).
    pub fn max_ffn(&self) -> usize {
        self.ffn_per_layer.iter().copied().max().unwrap_or(0)
    }

    /// Max Q projection width across all layers (`n_heads × head_dim`).
    pub fn max_q_width(&self) -> usize {
        let _max_kv = self.kv_heads_per_layer.iter().copied().max().unwrap_or(0);
        let max_dim = self.full_head_dim.max(self.swa_head_dim);
        // Q width is the same for every layer (only head_dim varies);
        // use n_heads × max_dim as an upper bound.
        self.n_heads * max_dim
    }

    /// Max KV projection width across all layers (drives scratch sizing
    /// for K/V intermediate buffers). Uses per-layer kv_heads × head_dim.
    pub fn max_kv_width(&self) -> usize {
        self.kv_heads_per_layer
            .iter()
            .copied()
            .enumerate()
            .map(|(i, kv)| kv * self.head_dim(i))
            .max()
            .unwrap_or(0)
    }

    /// Total per-layer token-embedding width (`layers × per_layer_width`).
    /// Zero when per-layer projection is disabled (12B).
    pub fn per_layer_all(&self) -> usize {
        self.layers * self.per_layer_width
    }

    /// `true` when the per-layer projection is enabled. 12B sets
    /// `per_layer_width = 0` and ships no `per_layer_*` tensors; E2B/E4B
    /// keep it on.
    pub fn use_per_layer_projection(&self) -> bool {
        self.per_layer_width > 0
    }

    /// `true` when this layer uses sliding-window attention, `false` for
    /// full attention. Matches the `sliding_window_pattern` metadata.
    pub fn is_swa(&self, layer: usize) -> bool {
        self.swa_pattern[layer]
    }

    /// Head dim used by this layer's attention (depends on SWA vs full).
    pub fn head_dim(&self, layer: usize) -> usize {
        if self.is_swa(layer) {
            self.swa_head_dim
        } else {
            self.full_head_dim
        }
    }

    /// Per-layer KV head count.
    pub fn kv_heads(&self, layer: usize) -> usize {
        self.kv_heads_per_layer[layer]
    }

    /// Q projection width for this layer (`n_heads × head_dim`).
    pub fn q_width(&self, layer: usize) -> usize {
        self.n_heads * self.head_dim(layer)
    }

    /// KV projection width for this layer (`kv_heads × head_dim`).
    pub fn kv_width(&self, layer: usize) -> usize {
        self.kv_heads(layer) * self.head_dim(layer)
    }

    pub fn from_source(source: &dyn TensorSource) -> Result<Self, String> {
        require_string(source, "general.architecture", "gemma4")?;
        require_string(source, "general.type", "model")?;

        let layers = read_u32(source, "gemma4.block_count")? as usize;

        // Allow either form (matches Qwen-style per-layer export vs the
        // scalar form llama.cpp uses for uniform FFN like E4B / 12B).
        let ffn_per_layer = match source.metadata("gemma4.feed_forward_length") {
            Some(MetaValue::Array(MetaValueType::Int32, values)) => {
                let mut per_layer = Vec::with_capacity(values.len());
                for v in values {
                    let MetaValue::Int32(n) = v else {
                        return Err(format!(
                            "Invalid metadata gemma4.feed_forward_length: expected Int32 array, got {v:?}"
                        ));
                    };
                    per_layer.push(
                        usize::try_from(*n).map_err(|_| {
                            format!("Invalid gemma4.feed_forward_length entry: {n}")
                        })?,
                    );
                }
                if per_layer.len() != layers {
                    return Err(format!(
                        "Inconsistent gemma4 metadata: feed_forward_length has {} entries, block_count is {layers}",
                        per_layer.len()
                    ));
                }
                per_layer
            }
            Some(MetaValue::Uint32(n)) => {
                let n = usize::try_from(*n)
                    .map_err(|_| format!("Invalid gemma4.feed_forward_length scalar: {n}"))?;
                vec![n; layers]
            }
            Some(MetaValue::Int32(n)) => {
                let n = usize::try_from(*n)
                    .map_err(|_| format!("Invalid gemma4.feed_forward_length scalar: {n}"))?;
                vec![n; layers]
            }
            Some(other) => {
                return Err(format!(
                    "Invalid metadata gemma4.feed_forward_length: expected array or scalar, got {other:?}"
                ));
            }
            None => return Err("Missing metadata: gemma4.feed_forward_length".into()),
        };

        let swa_pattern = match source.metadata("gemma4.attention.sliding_window_pattern") {
            Some(MetaValue::Array(MetaValueType::Bool, values)) => {
                let mut pattern = Vec::with_capacity(values.len());
                for v in values {
                    let MetaValue::Bool(b) = v else {
                        return Err(format!(
                            "Invalid metadata gemma4.attention.sliding_window_pattern: expected Bool array, got {v:?}"
                        ));
                    };
                    pattern.push(*b);
                }
                if pattern.len() != layers {
                    return Err(format!(
                        "Invalid metadata gemma4.attention.sliding_window_pattern: expected {layers} entries, got {}",
                        pattern.len()
                    ));
                }
                pattern
            }
            Some(other) => {
                return Err(format!(
                    "Invalid metadata gemma4.attention.sliding_window_pattern: expected Bool array, got {other:?}"
                ));
            }
            None => return Err("Missing metadata: gemma4.attention.sliding_window_pattern".into()),
        };

        // Read context_length from metadata. Gemma 4 ships with 131072
        // (E2B/E4B) or 262144 (12B); both are accepted here. Reject only
        // zero or implausibly large values that would indicate a corrupt
        // read. Mirrors the qwen35 approach (loader trusts GGUF).
        let n_ctx = read_u32(source, "gemma4.context_length")? as usize;
        if n_ctx == 0 || n_ctx > MAX_CONTEXT {
            return Err(format!(
                "Invalid metadata gemma4.context_length: {n_ctx} is outside (0, {MAX_CONTEXT}]"
            ));
        }
        let embd = read_u32(source, "gemma4.embedding_length")? as usize;

        // Query head count is uniform per model: 8 (E2B/E4B) or 16 (12B).
        let n_heads = read_u32(source, "gemma4.attention.head_count")? as usize;
        if n_heads == 0 {
            return Err("Invalid metadata gemma4.attention.head_count: 0".into());
        }

        // KV head count is either scalar (E2B/E4B: uniform 1) or
        // per-layer Int32 array (12B: heterogeneous 1/8). The Int32 form
        // is also accepted as a scalar singleton.
        let kv_heads_per_layer = match source.metadata("gemma4.attention.head_count_kv") {
            Some(MetaValue::Array(MetaValueType::Int32, values)) => {
                let mut per_layer = Vec::with_capacity(values.len());
                for v in values {
                    let MetaValue::Int32(n) = v else {
                        return Err(format!(
                            "Invalid metadata gemma4.attention.head_count_kv: expected Int32 array, got {v:?}"
                        ));
                    };
                    let n = usize::try_from(*n).map_err(|_| {
                        format!("Invalid gemma4.attention.head_count_kv entry: {n}")
                    })?;
                    per_layer.push(n);
                }
                if per_layer.len() != layers {
                    return Err(format!(
                        "Inconsistent gemma4 metadata: head_count_kv has {} entries, block_count is {layers}",
                        per_layer.len()
                    ));
                }
                per_layer
            }
            Some(MetaValue::Uint32(n)) => {
                let n = usize::try_from(*n)
                    .map_err(|_| format!("Invalid gemma4.attention.head_count_kv scalar: {n}"))?;
                vec![n; layers]
            }
            Some(MetaValue::Int32(n)) => {
                let n = usize::try_from(*n)
                    .map_err(|_| format!("Invalid gemma4.attention.head_count_kv scalar: {n}"))?;
                vec![n; layers]
            }
            Some(other) => {
                return Err(format!(
                    "Invalid metadata gemma4.attention.head_count_kv: expected array or scalar, got {other:?}"
                ));
            }
            None => return Err("Missing metadata: gemma4.attention.head_count_kv".into()),
        };

        let rope_freq_base = read_f32(source, "gemma4.rope.freq_base")?;
        let rope_freq_base_swa = read_f32(source, "gemma4.rope.freq_base_swa")?;
        let logit_softcap = read_f32(source, "gemma4.final_logit_softcapping")?;
        let sliding_window = read_u32(source, "gemma4.attention.sliding_window")? as usize;
        let shared_kv_layers = read_u32(source, "gemma4.attention.shared_kv_layers")? as usize;

        // Per-layer projection width. 0 means the projection is disabled
        // and the per_layer_* tensors are absent (12B). Anything else is
        // the per-layer embedding dim (E2B/E4B: 256).
        let per_layer_width = read_u32(source, "gemma4.embedding_length_per_layer_input")? as usize;

        // head dims. E2B/E4B full=512, swa=256. 12B same.
        let full_head_dim = read_u32(source, "gemma4.attention.key_length")? as usize;
        let value_length = read_u32(source, "gemma4.attention.value_length")? as usize;
        if value_length != full_head_dim {
            return Err(format!(
                "Inconsistent gemma4 metadata: key_length={full_head_dim}, value_length={value_length}"
            ));
        }
        let swa_head_dim = read_u32(source, "gemma4.attention.key_length_swa")? as usize;
        let value_length_swa = read_u32(source, "gemma4.attention.value_length_swa")? as usize;
        if value_length_swa != swa_head_dim {
            return Err(format!(
                "Inconsistent gemma4 metadata: key_length_swa={swa_head_dim}, value_length_swa={value_length_swa}"
            ));
        }
        let rope_dim_count = read_u32(source, "gemma4.rope.dimension_count")? as usize;
        if rope_dim_count != full_head_dim {
            return Err(format!(
                "Inconsistent gemma4 metadata: rope.dimension_count={rope_dim_count}, key_length={full_head_dim}"
            ));
        }
        let rope_dim_count_swa = read_u32(source, "gemma4.rope.dimension_count_swa")? as usize;
        if rope_dim_count_swa != swa_head_dim {
            return Err(format!(
                "Inconsistent gemma4 metadata: rope.dimension_count_swa={rope_dim_count_swa}, key_length_swa={swa_head_dim}"
            ));
        }

        // Sanity: layer KV heads must divide n_heads evenly so GQA
        // group_size is integer. (Always true for E2B's 8/1 and 12B's
        // 16/8 / 16/1, but rejected early if a corrupt export breaks it.)
        for (i, &kv) in kv_heads_per_layer.iter().enumerate() {
            if kv == 0 {
                return Err(format!(
                    "Invalid gemma4 metadata: head_count_kv[{i}] = 0 (must be > 0)"
                ));
            }
            if !n_heads.is_multiple_of(kv) {
                return Err(format!(
                    "Invalid gemma4 metadata: head_count ({n_heads}) is not a multiple of head_count_kv[{i}] ({kv})"
                ));
            }
        }

        require_string(source, "tokenizer.ggml.model", "gemma4")?;
        require_gemma4_token_table(source)?;

        let cfg = Self {
            layers,
            embd,
            n_heads,
            kv_heads_per_layer,
            vocab: VOCAB,
            full_head_dim,
            swa_head_dim,
            shared_kv_layers,
            sliding_window,
            logit_softcap,
            ffn_per_layer,
            swa_pattern,
            per_layer_width,
            n_ctx,
            rope_freq_base,
            rope_freq_base_swa,
        };

        cfg.validate_tensors(source)?;
        Ok(cfg)
    }

    /// Walk every tensor the trunk reads and confirm its shape / dtype
    /// against the resolved config. Per-layer projection tensors are
    /// only required when `per_layer_width > 0`. Per-layer attn_v is
    /// only required for SWA layers (12B shares K with V for full-attn
    /// MQA layers; E2B/E4B store V for every layer).
    fn validate_tensors(&self, source: &dyn TensorSource) -> Result<(), String> {
        let per_layer_all = self.per_layer_all();

        require_tensor(
            source,
            "output_norm.weight",
            &[self.embd as u64],
            GGMLType::F32,
        )?;
        // rope_freqs is sized to the larger of the two head_dims (full
        // covers the SWA case as a prefix). E2B/E4B use
        // full_head_dim/2; 12B uses swa_head_dim/2 (it stores only the
        // shorter SWA table because full-attn RoPE uses a different
        // table -- but inspect: the 12B dump shows rope_freqs=[256],
        // and 12B's full head_dim is 512. The RoPE table indexing for
        // the 12B full layers reads from a 512-entry table that is
        // built from this 256-entry base by symmetry. Accept either
        // [full/2] or [swa/2] for rope_freqs as the GGUF export
        // convention varies.)
        let rope_freqs_dim = (self.full_head_dim / 2) as u64;
        let rope_freqs_dim_swa = (self.swa_head_dim / 2) as u64;
        let rope_info = source
            .tensor_info("rope_freqs.weight")
            .ok_or_else(|| "Missing tensor: rope_freqs.weight".to_string())?;
        if rope_info.ggml_type != GGMLType::F32
            || (rope_info.dims.first().copied() != Some(rope_freqs_dim)
                && rope_info.dims.first().copied() != Some(rope_freqs_dim_swa))
        {
            return Err(format!(
                "Invalid tensor rope_freqs.weight: shape {:?} type {:?}; expected [{}] F32 or [{}] F32",
                rope_info.dims, rope_info.ggml_type, rope_freqs_dim, rope_freqs_dim_swa
            ));
        }
        if self.use_per_layer_projection() {
            require_tensor(
                source,
                "per_layer_model_proj.weight",
                &[self.embd as u64, per_layer_all as u64],
                GGMLType::BF16,
            )?;
            require_tensor(
                source,
                "per_layer_proj_norm.weight",
                &[self.per_layer_width as u64],
                GGMLType::F32,
            )?;
            require_tensor_any(
                source,
                "per_layer_token_embd.weight",
                &[per_layer_all as u64, VOCAB as u64],
                &[GGMLType::Q8_0, GGMLType::Q5K],
            )?;
        } else {
            // Verify the per-layer projection tensors are *absent*
            // (or, if present, just ignored). We don't reject them so
            // a future variant can opt back in by setting
            // per_layer_width > 0.
        }
        // Mixed-quants for token_embd. token_embd may also be Q4_K in
        // mixed-quants.
        require_tensor_any(
            source,
            "token_embd.weight",
            &[self.embd as u64, VOCAB as u64],
            &[GGMLType::Q8_0, GGMLType::Q4K],
        )?;

        for layer in 0..self.layers {
            let head_dim = self.head_dim(layer);
            let q_dim = self.n_heads * head_dim;
            let kv_dim = self.kv_heads(layer) * head_dim;
            let ffn = self.ffn_per_layer[layer];
            let prefix = format!("blk.{layer}");
            // Mixed-quants: matmul weights appear as Q4_K (gate/up/q/k
            // and most output) or Q6_K (v, down) per-layer; Q8_0 is
            // the baseline for 8-bit exports; Q4_0/Q5_0/Q4_1/Q5_1 for
            // Q4_0-quantized exports. K-quants are supported by all
            // downstream matmul kernels.
            let k_quant = [
                GGMLType::Q8_0,
                GGMLType::Q4K,
                GGMLType::Q6K,
                GGMLType::Q4_0,
                GGMLType::Q4_1,
                GGMLType::Q5_0,
                GGMLType::Q5_1,
            ];
            require_tensor_any(
                source,
                &format!("{prefix}.attn_k.weight"),
                &[self.embd as u64, kv_dim as u64],
                &k_quant,
            )?;
            // attn_k_norm is optional in some 12B exports when
            // kv_heads == 1; treat as optional.
            let _ = source.tensor_info(&format!("{prefix}.attn_k_norm.weight"));
            require_tensor(
                source,
                &format!("{prefix}.attn_norm.weight"),
                &[self.embd as u64],
                GGMLType::F32,
            )?;
            require_tensor_any(
                source,
                &format!("{prefix}.attn_output.weight"),
                &[q_dim as u64, self.embd as u64],
                &k_quant,
            )?;
            require_tensor_any(
                source,
                &format!("{prefix}.attn_q.weight"),
                &[self.embd as u64, q_dim as u64],
                &k_quant,
            )?;
            require_tensor(
                source,
                &format!("{prefix}.attn_q_norm.weight"),
                &[head_dim as u64],
                GGMLType::F32,
            )?;
            // attn_v is required for layers with kv_heads > 1. For
            // 12B full-attn (kv_heads=1) layers the export omits V and
            // the engine falls back to V := K (MQA sharing).
            if self.kv_heads(layer) > 1 {
                require_tensor_any(
                    source,
                    &format!("{prefix}.attn_v.weight"),
                    &[self.embd as u64, kv_dim as u64],
                    &k_quant,
                )?;
            } else {
                // Verify absence (informational; the runtime will reuse
                // K as V if V is missing).
                let _ = source.tensor_info(&format!("{prefix}.attn_v.weight"));
            }
            require_tensor_any(
                source,
                &format!("{prefix}.ffn_down.weight"),
                &[ffn as u64, self.embd as u64],
                &k_quant,
            )?;
            require_tensor_any(
                source,
                &format!("{prefix}.ffn_gate.weight"),
                &[self.embd as u64, ffn as u64],
                &k_quant,
            )?;
            require_tensor(
                source,
                &format!("{prefix}.ffn_norm.weight"),
                &[self.embd as u64],
                GGMLType::F32,
            )?;
            require_tensor_any(
                source,
                &format!("{prefix}.ffn_up.weight"),
                &[self.embd as u64, ffn as u64],
                &k_quant,
            )?;
            if self.use_per_layer_projection() {
                require_tensor(
                    source,
                    &format!("{prefix}.inp_gate.weight"),
                    &[self.embd as u64, self.per_layer_width as u64],
                    GGMLType::F32,
                )?;
                require_tensor(
                    source,
                    &format!("{prefix}.layer_output_scale.weight"),
                    &[1],
                    GGMLType::F32,
                )?;
                require_tensor(
                    source,
                    &format!("{prefix}.post_attention_norm.weight"),
                    &[self.embd as u64],
                    GGMLType::F32,
                )?;
                require_tensor(
                    source,
                    &format!("{prefix}.post_ffw_norm.weight"),
                    &[self.embd as u64],
                    GGMLType::F32,
                )?;
                require_tensor(
                    source,
                    &format!("{prefix}.post_norm.weight"),
                    &[self.embd as u64],
                    GGMLType::F32,
                )?;
                require_tensor(
                    source,
                    &format!("{prefix}.proj.weight"),
                    &[self.per_layer_width as u64, self.embd as u64],
                    GGMLType::F32,
                )?;
            } else {
                // Per-layer projection is disabled for 12B. Skip the
                // three per-layer tensors (inp_gate, proj, post_norm).
                // layer_output_scale, post_attention_norm and
                // post_ffw_norm are still required (they are part of
                // the standard residual path).
                require_tensor(
                    source,
                    &format!("{prefix}.layer_output_scale.weight"),
                    &[1],
                    GGMLType::F32,
                )?;
                require_tensor(
                    source,
                    &format!("{prefix}.post_attention_norm.weight"),
                    &[self.embd as u64],
                    GGMLType::F32,
                )?;
                require_tensor(
                    source,
                    &format!("{prefix}.post_ffw_norm.weight"),
                    &[self.embd as u64],
                    GGMLType::F32,
                )?;
            }
        }
        Ok(())
    }
}

/// Read a u32 metadata value (no equality check).
fn read_u32(source: &dyn TensorSource, key: &str) -> Result<u32, String> {
    match source.metadata(key) {
        Some(MetaValue::Uint32(v)) => Ok(*v),
        Some(other) => Err(format!(
            "Invalid metadata {key}: expected Uint32, got {other:?}"
        )),
        None => Err(format!("Missing metadata: {key}")),
    }
}

/// Read a f32 metadata value.
fn read_f32(source: &dyn TensorSource, key: &str) -> Result<f32, String> {
    match source.metadata(key) {
        Some(MetaValue::Float32(v)) => Ok(*v),
        Some(other) => Err(format!(
            "Invalid metadata {key}: expected Float32, got {other:?}"
        )),
        None => Err(format!("Missing metadata: {key}")),
    }
}
