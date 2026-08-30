//! LFM2-MoE model configuration derived from GGUF metadata (`lfm2moe.*` keys).

use crate::core::tensor::{MetaValue, TensorSource};

pub struct Lfm2MoeConfig {
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_embd_head_k: usize,
    pub n_embd_head_v: usize,
    /// Dense FFN width (`feed_forward_length`), used by the leading blocks.
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
    /// Total number of experts per MoE block.
    pub n_expert: usize,
    /// Experts routed per token.
    pub n_expert_used: usize,
    /// Per-expert FFN width (`expert_feed_forward_length`).
    pub n_ff_exp: usize,
    /// Blocks `[0, n_layer_dense_lead)` use the dense FFN; the rest are MoE.
    pub n_layer_dense_lead: usize,
    /// Router gating function: 1 = softmax, 2 = sigmoid (LFM2-8B-A1B ships 2).
    pub expert_gating_func: u32,
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

impl Lfm2MoeConfig {
    pub fn from_source<S: TensorSource + ?Sized>(source: &S) -> Result<Self, String> {
        const ARCH: &str = "lfm2moe";
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

        let n_embd = get_u32(&format!("{ARCH}.embedding_length"))? as usize;
        let n_layer = get_u32(&format!("{ARCH}.block_count"))? as usize;
        let n_head = get_u32(&format!("{ARCH}.attention.head_count"))? as usize;
        let n_ff = get_u32(&format!("{ARCH}.feed_forward_length"))? as usize;
        let n_ctx = get_u32(&format!("{ARCH}.context_length"))? as usize;
        let rope_freq_base = get_f32_opt(&format!("{ARCH}.rope.freq_base"), 1_000_000.0)?;
        let norm_eps = get_f32(&format!("{ARCH}.attention.layer_norm_rms_epsilon"))?;

        let n_embd_head_k = source
            .metadata(&format!("{ARCH}.attention.key_length"))
            .and_then(MetaValue::to_u64)
            .map(|v| v as usize)
            .unwrap_or(n_embd / n_head);
        let n_embd_head_v = source
            .metadata(&format!("{ARCH}.attention.value_length"))
            .and_then(MetaValue::to_u64)
            .map(|v| v as usize)
            .unwrap_or(n_embd_head_k);

        let head_kv_arr =
            read_i32_array(source, &format!("{ARCH}.attention.head_count_kv"), n_layer)?;
        let n_head_kv_per_layer: Vec<usize> =
            head_kv_arr.iter().map(|&v| v.max(0) as usize).collect();

        let l_cache = get_u32(&format!("{ARCH}.shortconv.l_cache"))? as usize;
        let d_conv = l_cache.saturating_sub(1).max(1);

        let n_expert = get_u32(&format!("{ARCH}.expert_count"))? as usize;
        let n_expert_used = get_u32(&format!("{ARCH}.expert_used_count"))? as usize;
        let n_ff_exp = get_u32(&format!("{ARCH}.expert_feed_forward_length"))? as usize;
        let n_layer_dense_lead = get_u32(&format!("{ARCH}.leading_dense_block_count"))? as usize;
        let expert_gating_func = get_u32(&format!("{ARCH}.expert_gating_func"))?;

        let vocab_size = match get_u32(&format!("{ARCH}.vocab_size")) {
            Ok(value) => value as usize,
            Err(_) => source
                .metadata("tokenizer.ggml.tokens")
                .and_then(MetaValue::to_arr)
                .map(Vec::len)
                .unwrap_or(0),
        };

        Ok(Lfm2MoeConfig {
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
            n_expert,
            n_expert_used,
            n_ff_exp,
            n_layer_dense_lead,
            expert_gating_func,
        })
    }
}
