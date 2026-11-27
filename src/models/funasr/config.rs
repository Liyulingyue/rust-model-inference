//! Fun-ASR-Nano encoder configuration parsed from GGUF metadata.

use crate::core::tensor::{MetaValue, TensorSource};

#[derive(Debug, Clone, Copy)]
pub struct FunAsrConfig {
    pub input_size: usize,
    pub output_size: usize,
    pub attention_heads: usize,
    pub linear_units: usize,
    pub num_blocks: usize,
    pub tp_blocks: usize,
    pub kernel_size: usize,
    pub sanm_shift: usize,
    pub adp_llm_dim: usize,
    pub adp_encoder_dim: usize,
    pub adp_ffn_dim: usize,
    pub adp_n_layer: usize,
    pub adp_attention_heads: usize,
    pub adp_downsample_rate: usize,
    pub frontend_n_mels: usize,
    pub frontend_lfr_m: usize,
    pub frontend_lfr_n: usize,
}

impl Default for FunAsrConfig {
    fn default() -> Self {
        Self {
            input_size: 560,
            output_size: 512,
            attention_heads: 4,
            linear_units: 2048,
            num_blocks: 50,
            tp_blocks: 20,
            kernel_size: 11,
            sanm_shift: 0,
            adp_llm_dim: 1024,
            adp_encoder_dim: 512,
            adp_ffn_dim: 2048,
            adp_n_layer: 2,
            adp_attention_heads: 8,
            adp_downsample_rate: 1,
            frontend_n_mels: 80,
            frontend_lfr_m: 7,
            frontend_lfr_n: 6,
        }
    }
}

impl FunAsrConfig {
    pub fn from_source(source: &dyn TensorSource) -> Result<Self, String> {
        let arch = source
            .metadata("general.architecture")
            .and_then(MetaValue::to_string_val)
            .unwrap_or_default();
        if arch != crate::models::funasr::ENCODER_ARCH {
            return Err(format!(
                "expected architecture {:?}, got {arch:?}",
                crate::models::funasr::ENCODER_ARCH
            ));
        }
        let cfg = Self::default();
        let read = |key: &str, default: usize| {
            source
                .metadata(key)
                .and_then(MetaValue::to_u64)
                .map(|v| v as usize)
                .unwrap_or(default)
        };
        Ok(Self {
            input_size: read("funasr.enc.input_size", cfg.input_size),
            output_size: read("funasr.enc.output_size", cfg.output_size),
            attention_heads: read("funasr.enc.attention_heads", cfg.attention_heads),
            linear_units: read("funasr.enc.linear_units", cfg.linear_units),
            num_blocks: read("funasr.enc.num_blocks", cfg.num_blocks),
            tp_blocks: read("funasr.enc.tp_blocks", cfg.tp_blocks),
            kernel_size: read("funasr.enc.kernel_size", cfg.kernel_size),
            sanm_shift: read("funasr.enc.sanm_shfit", cfg.sanm_shift),
            adp_llm_dim: read("funasr.adp.llm_dim", cfg.adp_llm_dim),
            adp_encoder_dim: read("funasr.adp.encoder_dim", cfg.adp_encoder_dim),
            adp_ffn_dim: read("funasr.adp.ffn_dim", cfg.adp_ffn_dim),
            adp_n_layer: read("funasr.adp.n_layer", cfg.adp_n_layer),
            adp_attention_heads: read("funasr.adp.attention_heads", cfg.adp_attention_heads),
            adp_downsample_rate: read("funasr.adp.downsample_rate", cfg.adp_downsample_rate),
            frontend_n_mels: read("funasr.frontend.n_mels", cfg.frontend_n_mels),
            frontend_lfr_m: read("funasr.frontend.lfr_m", cfg.frontend_lfr_m),
            frontend_lfr_n: read("funasr.frontend.lfr_n", cfg.frontend_lfr_n),
        })
    }
}
