use super::super::contract::{
    require_gemma4_token_table, require_string, require_tensor, require_tensor_any,
};
use crate::core::tensor::{GGMLType, MetaValue, MetaValueType, TensorSource};

/// Architecture constants shared across all Gemma 4 variants (E2B, E4B,
/// future revisions). Per-model dimensions live on `Gemma4Config`.
pub(super) const HEADS: usize = 8;
pub(super) const FULL_HEAD_DIM: usize = 512;
pub(super) const SWA_HEAD_DIM: usize = 256;
pub(super) const PER_LAYER: usize = 256;
pub(super) const VOCAB: usize = 262_144;
pub(super) const CONTEXT: usize = 131_072;
pub(super) const EPS: f32 = 1e-6;

#[derive(Debug, Clone, PartialEq)]
pub struct Gemma4Config {
    pub layers: usize,
    pub embd: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub vocab: usize,
    pub full_head_dim: usize,
    pub swa_head_dim: usize,
    pub shared_kv_layers: usize,
    pub per_layer_width: usize,
    pub sliding_window: usize,
    pub logit_softcap: f32,
    /// Per-layer FFN width. Length == `layers`. E2B is heterogeneous
    /// ([6144; 15] ++ [12288; 20]); E4B is uniform (10240 × 42).
    pub ffn_per_layer: Vec<usize>,
    /// Per-layer sliding-window boolean. `true` = SWA, `false` = full
    /// attention. Length == `layers`.
    pub swa_pattern: Vec<bool>,
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

    /// Total per-layer token-embedding width (`layers × per_layer_width`).
    pub fn per_layer_all(&self) -> usize {
        self.layers * self.per_layer_width
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

    pub fn from_source(source: &dyn TensorSource) -> Result<Self, String> {
        require_string(source, "general.architecture", "gemma4")?;
        require_string(source, "general.type", "model")?;

        let layers = read_u32(source, "gemma4.block_count")? as usize;

        // Allow either form (matches Qwen-style per-layer export vs the
        // scalar form llama.cpp uses for uniform FFN like E4B).
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

        // Validate-and-read combined calls: read first to validate equality.
        check_u32(source, "gemma4.context_length", CONTEXT as u32)?;
        let embd = read_u32(source, "gemma4.embedding_length")? as usize;
        check_u32(source, "gemma4.attention.head_count", HEADS as u32)?;
        let kv_heads = read_u32(source, "gemma4.attention.head_count_kv")? as usize;
        check_f32(source, "gemma4.rope.freq_base", 1_000_000.0)?;
        check_f32(source, "gemma4.rope.freq_base_swa", 10_000.0)?;
        check_f32(source, "gemma4.attention.layer_norm_rms_epsilon", EPS)?;
        check_u32(source, "gemma4.attention.key_length", FULL_HEAD_DIM as u32)?;
        check_u32(
            source,
            "gemma4.attention.value_length",
            FULL_HEAD_DIM as u32,
        )?;
        let logit_softcap = read_f32(source, "gemma4.final_logit_softcapping")?;
        let sliding_window = read_u32(source, "gemma4.attention.sliding_window")? as usize;
        let shared_kv_layers = read_u32(source, "gemma4.attention.shared_kv_layers")? as usize;
        check_u32(
            source,
            "gemma4.embedding_length_per_layer_input",
            PER_LAYER as u32,
        )?;
        check_u32(
            source,
            "gemma4.attention.key_length_swa",
            SWA_HEAD_DIM as u32,
        )?;
        check_u32(
            source,
            "gemma4.attention.value_length_swa",
            SWA_HEAD_DIM as u32,
        )?;
        check_u32(source, "gemma4.rope.dimension_count", FULL_HEAD_DIM as u32)?;
        check_u32(
            source,
            "gemma4.rope.dimension_count_swa",
            SWA_HEAD_DIM as u32,
        )?;
        require_string(source, "tokenizer.ggml.model", "gemma4")?;
        require_gemma4_token_table(source)?;

        let per_layer_all = layers * PER_LAYER;

        for (name, dims, ty) in [
            ("output_norm.weight", &[embd as u64][..], GGMLType::F32),
            (
                "per_layer_model_proj.weight",
                &[embd as u64, per_layer_all as u64][..],
                GGMLType::BF16,
            ),
            (
                "per_layer_proj_norm.weight",
                &[PER_LAYER as u64][..],
                GGMLType::F32,
            ),
            (
                "rope_freqs.weight",
                &[(FULL_HEAD_DIM / 2) as u64][..],
                GGMLType::F32,
            ),
        ] {
            require_tensor(source, name, dims, ty)?;
        }
        // Mixed-quants like Q4_K_M use K-quants here (per_layer_token_embd is
        // typically Q5_K, token_embd is typically Q4_K). All three variants
        // have a working `embedding_lookup`, so accept either.
        require_tensor_any(
            source,
            "per_layer_token_embd.weight",
            &[per_layer_all as u64, VOCAB as u64][..],
            &[GGMLType::Q8_0, GGMLType::Q5K],
        )?;
        // token_embd may also be Q4_K in mixed-quants.
        require_tensor_any(
            source,
            "token_embd.weight",
            &[embd as u64, VOCAB as u64][..],
            &[GGMLType::Q8_0, GGMLType::Q4K],
        )?;
        for layer in 0..layers {
            let head_dim = if swa_pattern[layer] {
                SWA_HEAD_DIM
            } else {
                FULL_HEAD_DIM
            };
            let kv_dim = kv_heads * head_dim;
            let q_dim = HEADS * head_dim;
            let ffn = ffn_per_layer[layer];
            let prefix = format!("blk.{layer}");
            // Mixed-quants split: matmul weights appear as Q4_K (gate/up/q/k
            // and most output) or Q6_K (v, down) per-layer; Q8_0 is the
            // baseline for 8-bit exports; E4B Q4_0 exports use plain Q4_0
            // for every tensor. K-quants are supported by all downstream
            // matmul kernels.
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
                &[embd as u64, kv_dim as u64],
                &k_quant,
            )?;
            require_tensor(
                source,
                &format!("{prefix}.attn_k_norm.weight"),
                &[head_dim as u64],
                GGMLType::F32,
            )?;
            require_tensor(
                source,
                &format!("{prefix}.attn_norm.weight"),
                &[embd as u64],
                GGMLType::F32,
            )?;
            require_tensor_any(
                source,
                &format!("{prefix}.attn_output.weight"),
                &[q_dim as u64, embd as u64],
                &k_quant,
            )?;
            require_tensor_any(
                source,
                &format!("{prefix}.attn_q.weight"),
                &[embd as u64, q_dim as u64],
                &k_quant,
            )?;
            require_tensor(
                source,
                &format!("{prefix}.attn_q_norm.weight"),
                &[head_dim as u64],
                GGMLType::F32,
            )?;
            require_tensor_any(
                source,
                &format!("{prefix}.attn_v.weight"),
                &[embd as u64, kv_dim as u64],
                &k_quant,
            )?;
            require_tensor_any(
                source,
                &format!("{prefix}.ffn_down.weight"),
                &[ffn as u64, embd as u64],
                &k_quant,
            )?;
            require_tensor_any(
                source,
                &format!("{prefix}.ffn_gate.weight"),
                &[embd as u64, ffn as u64],
                &k_quant,
            )?;
            require_tensor(
                source,
                &format!("{prefix}.ffn_norm.weight"),
                &[embd as u64],
                GGMLType::F32,
            )?;
            require_tensor_any(
                source,
                &format!("{prefix}.ffn_up.weight"),
                &[embd as u64, ffn as u64],
                &k_quant,
            )?;
            require_tensor(
                source,
                &format!("{prefix}.inp_gate.weight"),
                &[embd as u64, PER_LAYER as u64],
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
                &[embd as u64],
                GGMLType::F32,
            )?;
            require_tensor(
                source,
                &format!("{prefix}.post_ffw_norm.weight"),
                &[embd as u64],
                GGMLType::F32,
            )?;
            require_tensor(
                source,
                &format!("{prefix}.post_norm.weight"),
                &[embd as u64],
                GGMLType::F32,
            )?;
            require_tensor(
                source,
                &format!("{prefix}.proj.weight"),
                &[PER_LAYER as u64, embd as u64],
                GGMLType::F32,
            )?;
        }

        Ok(Self {
            layers,
            embd,
            heads: HEADS,
            kv_heads,
            vocab: VOCAB,
            full_head_dim: FULL_HEAD_DIM,
            swa_head_dim: SWA_HEAD_DIM,
            shared_kv_layers,
            per_layer_width: PER_LAYER,
            sliding_window,
            logit_softcap,
            ffn_per_layer,
            swa_pattern,
        })
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

/// Validate a u32 metadata value against an expected constant.
fn check_u32(source: &dyn TensorSource, key: &str, expected: u32) -> Result<(), String> {
    let actual = read_u32(source, key)?;
    if actual != expected {
        return Err(format!(
            "Invalid metadata {key}: expected uint32 {expected}, got {actual}"
        ));
    }
    Ok(())
}

/// Validate a f32 metadata value against an expected constant (bit-exact).
fn check_f32(source: &dyn TensorSource, key: &str, expected: f32) -> Result<(), String> {
    let actual = read_f32(source, key)?;
    if actual.to_bits() != expected.to_bits() {
        return Err(format!(
            "Invalid metadata {key}: expected float32 {expected}, got {actual}"
        ));
    }
    Ok(())
}
