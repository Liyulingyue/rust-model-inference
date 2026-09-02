//! Spark 2.5 (Xunfei Spark 2.5) model configuration.
//!
//! Parsed from GGUF metadata with the `spark2_5.` prefix.
//! See `references/XFllama.cpp/src/models/spark2_5.cpp` for reference.

use crate::core::tensor::{MetaValue, TensorSource};

pub struct SparkConfig {
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub n_embd_head_k: usize,
    pub n_embd_head_v: usize,
    pub n_ff: usize,
    pub vocab: usize,
    pub n_ctx: usize,
    pub eps: f32,
    pub freq_base_full: f32,
    pub freq_base_swa: f32,
    pub n_rot_full: usize,
    pub n_rot_swa: usize,
    pub sliding_window: usize,
    /// Per-layer flag: `true` if the layer uses sliding-window attention.
    /// Pattern length must equal `n_layer`. Cycle is "3-swa + 1-full" for the 1.7B variant.
    pub is_swa: Vec<bool>,
}

fn read_u32<S: TensorSource + ?Sized>(source: &S, key: &str) -> Result<u32, String> {
    source
        .metadata(key)
        .and_then(MetaValue::to_u64)
        .and_then(|v| u32::try_from(v).ok())
        .ok_or_else(|| format!("Missing metadata: {key}"))
}

fn read_f32<S: TensorSource + ?Sized>(source: &S, key: &str) -> Result<f32, String> {
    source
        .metadata(key)
        .and_then(MetaValue::to_f64)
        .map(|v| v as f32)
        .ok_or_else(|| format!("Missing metadata: {key}"))
}

fn read_bool_array<S: TensorSource + ?Sized>(
    source: &S,
    key: &str,
    expected_len: usize,
) -> Result<Vec<bool>, String> {
    let value = source
        .metadata(key)
        .ok_or_else(|| format!("Missing metadata: {key}"))?;
    let MetaValue::Array(_, items) = value else {
        return Err(format!("{key} is not an array"));
    };
    if items.len() != expected_len {
        return Err(format!(
            "{key} expected {expected_len} entries, got {}",
            items.len()
        ));
    }
    items
        .iter()
        .map(|v| match v {
            MetaValue::Bool(b) => Ok(*b),
            _ => Err(format!("{key} has non-bool entries")),
        })
        .collect()
}

impl SparkConfig {
    pub fn from_source<S: TensorSource + ?Sized>(source: &S) -> Result<Self, String> {
        let arch = source
            .metadata("general.architecture")
            .and_then(MetaValue::to_string_val)
            .ok_or_else(|| "Missing general.architecture".to_string())?;
        if arch != "spark2_5" {
            return Err(format!("Unsupported architecture for SparkConfig: {arch}"));
        }

        let n_embd = read_u32(source, "spark2_5.embedding_length")? as usize;
        let n_layer = read_u32(source, "spark2_5.block_count")? as usize;
        let n_head = read_u32(source, "spark2_5.attention.head_count")? as usize;
        let n_head_kv = read_u32(source, "spark2_5.attention.head_count_kv")? as usize;
        let n_embd_head_k = read_u32(source, "spark2_5.attention.key_length")? as usize;
        let n_embd_head_v = read_u32(source, "spark2_5.attention.value_length")? as usize;
        let n_ff = read_u32(source, "spark2_5.feed_forward_length")? as usize;
        let vocab = read_u32(source, "spark2_5.vocab_size")? as usize;
        let n_ctx = read_u32(source, "spark2_5.context_length")? as usize;
        let eps = read_f32(source, "spark2_5.attention.layer_norm_rms_epsilon")?;
        let freq_base_full = read_f32(source, "spark2_5.rope.freq_base")?;
        let freq_base_swa = read_f32(source, "spark2_5.rope.freq_base_swa")?;
        let n_rot_full = read_u32(source, "spark2_5.rope.dimension_count")? as usize;
        let n_rot_swa = read_u32(source, "spark2_5.rope.dimension_count_swa")? as usize;
        let sliding_window = read_u32(source, "spark2_5.attention.sliding_window")? as usize;
        let is_swa = read_bool_array(
            source,
            "spark2_5.attention.sliding_window_pattern",
            n_layer,
        )?;

        if n_embd_head_k != n_embd_head_v {
            return Err(format!(
                "spark2_5 requires n_embd_head_k == n_embd_head_v (got {n_embd_head_k} vs {n_embd_head_v})"
            ));
        }

        Ok(Self {
            n_embd,
            n_layer,
            n_head,
            n_head_kv,
            n_embd_head_k,
            n_embd_head_v,
            n_ff,
            vocab,
            n_ctx,
            eps,
            freq_base_full,
            freq_base_swa,
            n_rot_full,
            n_rot_swa,
            sliding_window,
            is_swa,
        })
    }

    pub fn n_embd_q(&self) -> usize {
        self.n_head * self.n_embd_head_k
    }

    pub fn n_embd_kv(&self) -> usize {
        self.n_head_kv * self.n_embd_head_k
    }

    pub fn n_embd_qkv(&self) -> usize {
        self.n_embd_q() + 2 * self.n_embd_kv()
    }

    pub fn freq_base_for(&self, layer: usize) -> f32 {
        if self.is_swa[layer] {
            self.freq_base_swa
        } else {
            self.freq_base_full
        }
    }

    pub fn n_rot_for(&self, layer: usize) -> usize {
        if self.is_swa[layer] {
            self.n_rot_swa
        } else {
            self.n_rot_full
        }
    }
}