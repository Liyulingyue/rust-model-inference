//! Falcon-H1 configuration.
//!
//! Falcon-H1 (tiiuae) is a parallel hybrid architecture: every layer runs a
//! GQA attention branch and a Mamba2 SSM branch on the same normed input and
//! sums both outputs into the residual. Reference: llama.cpp
//! `src/models/falcon-h1.cpp` @ `171e8846b4af9766c354064cb776cb34a50f053f`.

use crate::core::tensor::{MetaValue, TensorSource};

#[derive(Debug, Clone)]
pub struct FalconH1Config {
    pub architecture: String,
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub n_embd_head_k: usize,
    pub n_embd_head_v: usize,
    pub n_ff: usize,
    pub n_ctx: usize,
    pub vocab_size: usize,
    pub rope_freq_base: f32,
    pub norm_eps: f32,
    pub ssm_conv_kernel: usize,
    pub ssm_state_size: usize,
    pub ssm_group_count: usize,
    pub ssm_inner_size: usize,
    pub ssm_time_step_rank: usize,
}

impl FalconH1Config {
    /// Number of SSM heads. Canonical Mamba2 uses one dt per head, and the
    /// GGUF stores `ssm_dt.bias.shape == (time_step_rank,)`, so
    /// n_ssm_head == ssm_time_step_rank (llama.cpp: `hparams.ssm_dt_rank`).
    pub fn ssm_n_head(&self) -> usize {
        self.ssm_time_step_rank
    }

    /// Per-head channel count: d_inner / n_head.
    pub fn ssm_headdim(&self) -> usize {
        self.ssm_inner_size / self.ssm_n_head()
    }

    /// Input width of the causal conv1d: x plus B and C.
    pub fn ssm_conv_cols(&self) -> usize {
        self.ssm_inner_size + 2 * self.ssm_group_count * self.ssm_state_size
    }

    /// Total width of the fused `ssm_in` projection:
    /// z + xBC + dt = d_inner + conv_cols + n_head.
    pub fn ssm_in_proj_dim(&self) -> usize {
        2 * self.ssm_inner_size + 2 * self.ssm_group_count * self.ssm_state_size
            + self.ssm_time_step_rank
    }
}

impl FalconH1Config {
    pub fn from_source(source: &dyn TensorSource) -> Result<Self, String> {
        let architecture = source
            .metadata("general.architecture")
            .and_then(MetaValue::to_string_val)
            .unwrap_or_default();
        if architecture != "falcon-h1" {
            return Err(format!(
                "Unsupported architecture for FalconH1Config: {architecture}"
            ));
        }
        let get_usize = |key: &str| -> Result<usize, String> {
            let v = source
                .metadata(key)
                .and_then(MetaValue::to_u64)
                .ok_or_else(|| format!("Missing metadata: {key}"))?;
            usize::try_from(v).map_err(|_| format!("Invalid metadata: {key} exceeds usize"))
        };
        let get_f32 = |key: &str, default: f32| -> Result<f32, String> {
            Ok(source
                .metadata(key)
                .and_then(MetaValue::to_f64)
                .map(|v| v as f32)
                .unwrap_or(default))
        };
        let config = Self {
            architecture: architecture.to_string(),
            n_embd: get_usize("falcon-h1.embedding_length")?,
            n_layer: get_usize("falcon-h1.block_count")?,
            n_head: get_usize("falcon-h1.attention.head_count")?,
            n_head_kv: get_usize("falcon-h1.attention.head_count_kv")?,
            n_embd_head_k: get_usize("falcon-h1.attention.key_length")?,
            n_embd_head_v: get_usize("falcon-h1.attention.value_length")?,
            n_ff: get_usize("falcon-h1.feed_forward_length")?,
            n_ctx: get_usize("falcon-h1.context_length")?,
            vocab_size: get_usize("falcon-h1.vocab_size")?,
            rope_freq_base: get_f32("falcon-h1.rope.freq_base", 10_000.0)?,
            norm_eps: get_f32("falcon-h1.attention.layer_norm_rms_epsilon", 1e-5)?,
            ssm_conv_kernel: get_usize("falcon-h1.ssm.conv_kernel")?,
            ssm_state_size: get_usize("falcon-h1.ssm.state_size")?,
            ssm_group_count: get_usize("falcon-h1.ssm.group_count")?,
            ssm_inner_size: get_usize("falcon-h1.ssm.inner_size")?,
            ssm_time_step_rank: get_usize("falcon-h1.ssm.time_step_rank")?,
        };
        if config.n_embd == 0
            || config.n_layer == 0
            || config.n_head == 0
            || config.n_head % config.n_head_kv != 0
            || config.n_embd_head_k == 0
            || config.n_embd_head_v == 0
            || config.n_ff == 0
            || config.ssm_conv_kernel < 2
            || config.ssm_group_count == 0
            || config.ssm_inner_size % config.ssm_group_count != 0
            || config.ssm_inner_size % config.ssm_time_step_rank != 0
            || config.ssm_time_step_rank % config.ssm_group_count != 0
        {
            return Err("Unsupported Falcon-H1 attention or SSM dimensions".into());
        }
        Ok(config)
    }

    pub fn head_dim(&self) -> usize {
        if self.n_head == 0 {
            0
        } else {
            self.n_embd / self.n_head
        }
    }
}
