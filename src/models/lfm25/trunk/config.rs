//! LFM2 model configuration derived from GGUF metadata.

use crate::core::tensor::{MetaValue, TensorSource};

pub struct Lfm25Config {
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_embd_head_k: usize,
    pub n_embd_head_v: usize,
    pub n_ff: usize,
    pub n_ctx: usize,
    pub vocab_size: usize,
    pub rope_freq_base: f32,
    pub norm_eps: f32,
    /// Per-layer head count KV (length = n_layer). A value of 0 marks the
    /// layer as recurrent (shortconv); non-zero marks attention.
    pub n_head_kv_per_layer: Vec<usize>,
    /// `d_conv = l_cache - 1` — the length of the conv state per channel.
    pub d_conv: usize,
    /// `l_cache` from metadata — the size of the conv kernel.
    pub l_cache: usize,
}

fn read_i32_array<S: TensorSource + ?Sized>(
    source: &S,
    key: &str,
    expected_len: usize,
) -> Result<Vec<i32>, String> {
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
            MetaValue::Int32(v) => *v,
            MetaValue::Uint32(v) => *v as i32,
            _ => return Err(format!("{key} has non-integer entries")),
        };
        out.push(v);
    }
    Ok(out)
}

impl Lfm25Config {
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
            Ok(source
                .metadata(key)
                .and_then(MetaValue::to_f64)
                .map(|v| v as f32)
                .unwrap_or(default))
        };

        let n_embd = get_u32("lfm2.embedding_length")? as usize;
        let n_layer = get_u32("lfm2.block_count")? as usize;
        let n_head = get_u32("lfm2.attention.head_count")? as usize;
        let n_ff = get_u32("lfm2.feed_forward_length")? as usize;
        let n_ctx = get_u32("lfm2.context_length")? as usize;
        let rope_freq_base = get_f32_opt("lfm2.rope.freq_base", 1_000_000.0)?;
        let norm_eps = get_f32("lfm2.attention.layer_norm_rms_epsilon")?;

        let n_embd_head_k = source
            .metadata("lfm2.attention.key_length")
            .and_then(MetaValue::to_u64)
            .map(|v| v as usize)
            .unwrap_or(n_embd / n_head);
        let n_embd_head_v = source
            .metadata("lfm2.attention.value_length")
            .and_then(MetaValue::to_u64)
            .map(|v| v as usize)
            .unwrap_or(n_embd_head_k);

        let head_kv_arr = read_i32_array(source, "lfm2.attention.head_count_kv", n_layer)?;
        let n_head_kv_per_layer: Vec<usize> =
            head_kv_arr.iter().map(|&v| v.max(0) as usize).collect();

        let l_cache = get_u32("lfm2.shortconv.l_cache")? as usize;
        let d_conv = l_cache.saturating_sub(1).max(1);

        let vocab_size = match get_u32("lfm2.vocab_size") {
            Ok(value) => value as usize,
            Err(_) => source
                .metadata("tokenizer.ggml.tokens")
                .and_then(MetaValue::to_arr)
                .map(Vec::len)
                .unwrap_or(0),
        };

        Ok(Lfm25Config {
            n_embd,
            n_layer,
            n_head,
            n_embd_head_k,
            n_embd_head_v,
            n_ff,
            n_ctx,
            vocab_size,
            rope_freq_base,
            norm_eps,
            n_head_kv_per_layer,
            d_conv,
            l_cache,
        })
    }
}
