//! LFM3 model configuration derived from GGUF metadata.

use crate::core::tensor::{MetaValue, TensorSource};

pub struct Lfm3Config {
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_embd_head_k: usize,
    pub n_embd_head_v: usize,
    pub n_ff: usize,
    pub n_ctx: usize,
    pub vocab_size: usize,
    pub rope_freq_base: f32,
    pub rope_freq_base_swa: f32,
    pub norm_eps: f32,
    /// Per-layer head count KV (length = n_layer). A value of 0 marks the
    /// layer as recurrent (shortconv); non-zero marks attention.
    pub n_head_kv_per_layer: Vec<usize>,
    /// `d_conv = l_cache - 1` — the length of the conv state per channel.
    pub d_conv: usize,
    /// `l_cache` from metadata — the size of the conv kernel.
    pub l_cache: usize,
    pub sliding_window: usize,
    pub sliding_window_pattern: Vec<bool>,
}

fn read_head_counts<S: TensorSource + ?Sized>(
    source: &S,
    key: &str,
    expected_len: usize,
) -> Result<Vec<usize>, String> {
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
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        let v = match item {
            MetaValue::Int32(v) => {
                usize::try_from(*v).map_err(|_| format!("{key} has negative entries"))?
            }
            MetaValue::Uint32(v) => *v as usize,
            _ => return Err(format!("{key} has non-integer entries")),
        };
        out.push(v);
    }
    Ok(out)
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
        .map(|item| match item {
            MetaValue::Bool(value) => Ok(*value),
            _ => Err(format!("{key} has non-boolean entries")),
        })
        .collect()
}

impl Lfm3Config {
    pub fn from_source<S: TensorSource + ?Sized>(source: &S) -> Result<Self, String> {
        let get_u32 = |key: &str| -> Result<u32, String> {
            source
                .metadata(key)
                .and_then(MetaValue::to_u64)
                .ok_or_else(|| format!("Missing metadata: {key}"))
                .and_then(|v| u32::try_from(v).map_err(|_| format!("{key} does not fit u32")))
        };
        let get_f32 = |key: &str| -> Result<f32, String> {
            source
                .metadata(key)
                .and_then(MetaValue::to_f64)
                .ok_or_else(|| format!("Missing metadata: {key}"))
                .map(|v| v as f32)
        };
        let get_f32_opt = |key: &str, default: f32| -> Result<f32, String> {
            if source.metadata(key).is_some() {
                get_f32(key)
            } else {
                Ok(default)
            }
        };

        if source
            .metadata("general.architecture")
            .and_then(MetaValue::to_string_val)
            != Some("lfm3")
        {
            return Err("expected general.architecture = lfm3".into());
        }
        let n_embd = get_u32("lfm3.embedding_length")? as usize;
        let n_layer = get_u32("lfm3.block_count")? as usize;
        let n_head = get_u32("lfm3.attention.head_count")? as usize;
        let n_ff = get_u32("lfm3.feed_forward_length")? as usize;
        let n_ctx = get_u32("lfm3.context_length")? as usize;
        let rope_freq_base = get_f32_opt("lfm3.rope.freq_base", 1_000_000.0)?;
        let rope_freq_base_swa = get_f32("lfm3.rope.freq_base_swa")?;
        let norm_eps = get_f32("lfm3.attention.layer_norm_rms_epsilon")?;

        if n_embd == 0 || n_layer == 0 || n_head == 0 || n_ff == 0 || n_ctx == 0 {
            return Err("lfm3 dimensions must be positive".into());
        }
        if [rope_freq_base, rope_freq_base_swa, norm_eps]
            .iter()
            .any(|v| !v.is_finite() || *v <= 0.0)
        {
            return Err("lfm3 RoPE bases and RMS epsilon must be positive and finite".into());
        }
        let n_embd_head_k = if source.metadata("lfm3.attention.key_length").is_some() {
            get_u32("lfm3.attention.key_length")? as usize
        } else {
            if n_embd % n_head != 0 {
                return Err(
                    "lfm3 hidden size must divide head count when key_length is absent".into(),
                );
            }
            n_embd / n_head
        };
        let n_embd_head_v = if source.metadata("lfm3.attention.value_length").is_some() {
            get_u32("lfm3.attention.value_length")? as usize
        } else {
            n_embd_head_k
        };

        if n_embd_head_k == 0 || n_embd_head_k % 2 != 0 || n_embd_head_v == 0 {
            return Err(
                "lfm3 attention head dimensions must be positive and key dimension even".into(),
            );
        }
        if n_embd_head_k != n_embd_head_v {
            return Err(format!(
                "lfm3 requires equal key/value head dimensions, got {n_embd_head_k}/{n_embd_head_v}"
            ));
        }

        if let Some(rot) = source.metadata("lfm3.rope.dimension_count") {
            if rot.to_u64() != Some(n_embd_head_k as u64) {
                return Err("lfm3 requires full-head rotary dimensions".into());
            }
        }
        let n_head_kv_per_layer =
            read_head_counts(source, "lfm3.attention.head_count_kv", n_layer)?;
        if n_head_kv_per_layer
            .iter()
            .any(|&heads| heads > n_head || (heads != 0 && n_head % heads != 0))
        {
            return Err("lfm3 per-layer KV head counts must divide attention head count".into());
        }

        let l_cache = get_u32("lfm3.shortconv.l_cache")? as usize;
        if l_cache < 2 {
            return Err("lfm3.shortconv.l_cache must be at least 2".into());
        }
        let d_conv = l_cache - 1;
        let sliding_window = get_u32("lfm3.attention.sliding_window")? as usize;
        if sliding_window == 0 {
            return Err("lfm3.attention.sliding_window must be positive".into());
        }
        let sliding_window_pattern =
            read_bool_array(source, "lfm3.attention.sliding_window_pattern", n_layer)?;
        if !sliding_window_pattern.iter().any(|&is_swa| is_swa) {
            return Err("lfm3 requires at least one sliding-window layer".into());
        }

        if sliding_window_pattern
            .iter()
            .zip(&n_head_kv_per_layer)
            .any(|(&swa, &heads)| swa && heads == 0)
        {
            return Err("lfm3 sliding-window layers must be attention layers".into());
        }
        let vocab_size = if source.metadata("lfm3.vocab_size").is_some() {
            get_u32("lfm3.vocab_size")? as usize
        } else {
            source
                .metadata("tokenizer.ggml.tokens")
                .and_then(MetaValue::to_arr)
                .map_or(0, Vec::len)
        };
        if vocab_size == 0 {
            return Err("lfm3 vocabulary must be nonempty".into());
        }
        for (a, b) in [
            (n_head, n_embd_head_k),
            (n_embd, l_cache),
            (n_embd, 3),
            (n_embd, n_ff),
            (n_embd, vocab_size),
        ] {
            a.checked_mul(b).ok_or("lfm3 dimension product overflow")?;
        }

        Ok(Lfm3Config {
            n_embd,
            n_layer,
            n_head,
            n_embd_head_k,
            n_embd_head_v,
            n_ff,
            n_ctx,
            vocab_size,
            rope_freq_base,
            rope_freq_base_swa,
            norm_eps,
            n_head_kv_per_layer,
            d_conv,
            l_cache,
            sliding_window,
            sliding_window_pattern,
        })
    }
}
