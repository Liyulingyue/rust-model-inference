//! Xing4.0-29B-A4B (TeleChat4) model configuration.
//!
//! Parses `xing4_0.*` GGUF metadata. The architecture combines four
//! mechanisms, each standard for the family this model belongs to:
//!
//! - **MLA** (multi-head latent attention, DeepSeek-style): queries are
//!   projected down to `q_lora_rank`, keys/values share a single
//!   `kv_lora_rank` latent per layer, and the KV cache stores the
//!   compressed latent instead of per-head K/V.
//! - **MoE** FFN: `expert_count` routed experts with `expert_used_count`
//!   active per token, plus `expert_shared_count` shared experts.
//!   `leading_dense_block_count` blocks use a dense FFN instead.
//! - **mHC** (hyper-connection): the residual stream is widened into
//!   `hyper_connection.count` parallel streams that are mixed into a
//!   single lane before each block and re-split afterwards.
//! - **MTP** (multi-token prediction): `nextn_predict_layers` extra
//!   blocks appended after the trunk. They are loaded but not executed
//!   by the plain decode loop.

use crate::core::tensor::{MetaValue, TensorSource};

pub const ARCH: &str = "xing4_0";

#[derive(Debug, Clone)]
pub struct Xing4Config {
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub n_ctx: usize,
    pub n_ff: usize,
    pub n_vocab: usize,
    pub norm_eps: f32,
    pub rope_freq_base: f32,
    pub rope_dim: usize,
    /// RoPE yarn scaling. Xing4.0 ships yarn with factor 64 over a 4096
    /// original context and 262144 trained tokens.
    pub rope_yarn_factor: f32,
    pub rope_yarn_orig_ctx: usize,
    pub rope_yarn_beta_fast: f32,
    pub rope_yarn_beta_slow: f32,
    pub rope_yarn_log_mult: f32,

    // MLA
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    /// full per-head key width `attention.key_length_mla`; the
    /// no-rotary part is this minus `n_embd_head_qk_rope`.
    pub n_embd_head_k: usize,
    /// per-head value width (`attention.value_length_mla`).
    pub n_embd_head_v: usize,
    /// rotary width appended to the KV latent (`rope.dimension_count`).
    pub n_embd_head_qk_rope: usize,

    // MoE
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub n_expert_shared: usize,
    pub n_ff_exp: usize,
    pub n_layer_dense_lead: usize,
    pub expert_gating_func: u32,
    pub expert_weights_norm: bool,
    pub expert_weights_scale: f32,

    // mHC
    pub hc_count: usize,
    pub hc_sinkhorn_iters: usize,
    pub hc_eps: f32,

    // MTP
    pub nextn_layers: usize,
}

impl Xing4Config {
    pub fn from_source<S: TensorSource + ?Sized>(source: &S) -> Result<Self, String> {
        let get_u32 = |key: &str| -> Result<usize, String> {
            source
                .metadata(key)
                .and_then(MetaValue::to_u64)
                .map(|v| v as usize)
                .ok_or_else(|| format!("Missing u32 metadata: {key}"))
        };
        let get_f32 = |key: &str| -> Result<f32, String> {
            source
                .metadata(key)
                .and_then(MetaValue::to_f64)
                .map(|v| v as f32)
                .ok_or_else(|| format!("Missing f32 metadata: {key}"))
        };
        let opt_f32 = |key: &str, default: f32| {
            source
                .metadata(key)
                .and_then(MetaValue::to_f64)
                .map(|v| v as f32)
                .unwrap_or(default)
        };

        let block_count = get_u32(&format!("{ARCH}.block_count"))?;
        let nextn_layers = get_u32(&format!("{ARCH}.nextn_predict_layers")).unwrap_or(0);
        let n_layer = block_count.saturating_sub(nextn_layers);

        Ok(Self {
            n_embd: get_u32(&format!("{ARCH}.embedding_length"))?,
            n_layer,
            n_head: get_u32(&format!("{ARCH}.attention.head_count"))?,
            // MLA caches one shared latent, so the "kv head" count is 1.
            n_head_kv: 1,
            n_ctx: get_u32(&format!("{ARCH}.context_length"))?,
            n_ff: get_u32(&format!("{ARCH}.feed_forward_length"))?,
            n_vocab: get_u32(&format!("{ARCH}.vocab_size"))?,
            norm_eps: get_f32(&format!("{ARCH}.attention.layer_norm_rms_epsilon"))?,
            rope_freq_base: get_f32(&format!("{ARCH}.rope.freq_base"))?,
            rope_dim: get_u32(&format!("{ARCH}.rope.dimension_count"))?,
            rope_yarn_factor: opt_f32(&format!("{ARCH}.rope.scaling.factor"), 1.0),
            rope_yarn_orig_ctx: get_u32(&format!("{ARCH}.rope.scaling.original_context_length"))
                .unwrap_or(4096),
            rope_yarn_beta_fast: opt_f32(&format!("{ARCH}.rope.scaling.yarn_beta_fast"), 32.0),
            rope_yarn_beta_slow: opt_f32(&format!("{ARCH}.rope.scaling.yarn_beta_slow"), 1.0),
            rope_yarn_log_mult: opt_f32(
                &format!("{ARCH}.rope.scaling.yarn_log_multiplier"),
                0.1,
            ),

            q_lora_rank: get_u32(&format!("{ARCH}.attention.q_lora_rank"))?,
            kv_lora_rank: get_u32(&format!("{ARCH}.attention.kv_lora_rank"))?,
            n_embd_head_k: get_u32(&format!("{ARCH}.attention.key_length_mla"))?,
            n_embd_head_v: get_u32(&format!("{ARCH}.attention.value_length_mla"))?,
            n_embd_head_qk_rope: get_u32(&format!("{ARCH}.rope.dimension_count"))?,

            n_expert: get_u32(&format!("{ARCH}.expert_count"))?,
            n_expert_used: get_u32(&format!("{ARCH}.expert_used_count"))?,
            n_expert_shared: get_u32(&format!("{ARCH}.expert_shared_count")).unwrap_or(0),
            n_ff_exp: get_u32(&format!("{ARCH}.expert_feed_forward_length"))?,
            n_layer_dense_lead: get_u32(&format!("{ARCH}.leading_dense_block_count"))
                .unwrap_or(0),
            expert_gating_func: source
                .metadata(&format!("{ARCH}.expert_gating_func"))
                .and_then(MetaValue::to_u64)
                .unwrap_or(2) as u32,
            expert_weights_norm: source
                .metadata(&format!("{ARCH}.expert_weights_norm"))
                .and_then(MetaValue::to_u64)
                .map(|v| v != 0)
                .unwrap_or(true),
            expert_weights_scale: opt_f32(&format!("{ARCH}.expert_weights_scale"), 1.0),

            hc_count: get_u32(&format!("{ARCH}.hyper_connection.count")).unwrap_or(1),
            hc_sinkhorn_iters: get_u32(&format!("{ARCH}.hyper_connection.sinkhorn_iterations"))
                .unwrap_or(1),
            hc_eps: opt_f32(&format!("{ARCH}.hyper_connection.epsilon"), 1e-6),

            nextn_layers,
        })
    }

    /// Width of one hyper-connection stream after flattening: the number
    /// of streams times the model width.
    pub fn hc_flat(&self) -> usize {
        self.hc_count * self.n_embd
    }

    /// Width of the mixture tensor one block's `hc_{attn,ffn}_fn` emits:
    /// `pre (hc) + post (hc) + comb (hc * hc)`.
    pub fn hc_mix_dim(&self) -> usize {
        (2 + self.hc_count) * self.hc_count
    }

    /// KV-cache width per token per layer: the compressed latent plus
    /// the rotary key part, shared across heads.
    pub fn kv_cache_width(&self) -> usize {
        self.kv_lora_rank + self.n_embd_head_qk_rope
    }

    /// Per-head query/key width emitted by `attn_q_b` (nope + rope).
    pub fn n_embd_head_qk(&self) -> usize {
        self.n_embd_head_k
    }

    /// Rotary-free part of the per-head key = `key_length_mla - rope`.
    pub fn n_embd_head_k_nope(&self) -> usize {
        self.n_embd_head_k - self.n_embd_head_qk_rope
    }

    /// True when layer `il` uses the dense FFN instead of the MoE.
    pub fn is_dense_ffn(&self, il: usize) -> bool {
        il < self.n_layer_dense_lead
    }
}
