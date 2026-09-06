//! VibeVoice ASR model configuration, derived from the mmproj GGUF metadata.

use crate::core::tensor::{MetaValue, MetaValueType, TensorSource};

#[derive(Debug, Clone)]
pub struct VibeVoiceAsrConfig {
    /// Audio sample rate the tokenizers expect (24000).
    pub sample_rate: usize,
    /// Audio samples per latent frame (3200 → 7.5 frames per second).
    pub compress_ratio: usize,
    /// Latent frames committed per streaming chunk (22 → 2.933 s).
    pub chunk_frames: usize,
    /// Extra lookahead latent frames per chunk (4 → 0.533 s).
    pub lookahead_frames: usize,
    /// LLM hidden width (3584).
    pub llm_hidden_size: usize,
    /// Acoustic tokenizer latent width (64).
    pub acoustic_vae_dim: usize,
    /// Semantic tokenizer latent width (128).
    pub semantic_vae_dim: usize,
    /// ConvNeXt stem filter count (32).
    pub n_filters: usize,
    /// Checkpoint encoder ratios in config order (reversed at runtime).
    pub ratios: Vec<usize>,
    /// ConvNeXt stage depths ([3, 3, 3, 3, 3, 3, 8]).
    pub depths: Vec<usize>,
    /// Block / downsample / stem kernel size (7).
    pub kernel_size: usize,
    /// Encoder head kernel size (7).
    pub last_kernel_size: usize,
    /// FFN expansion (4).
    pub ffn_expansion: usize,
    /// Causal-conv padding mode ("constant").
    pub pad_mode: String,
    /// Block mixer kind ("depthwise_conv").
    pub mixer_layer: String,
    /// ConvRMSNorm epsilon inside blocks (1e-5).
    pub layernorm_eps: f32,
    /// SpeechConnector RMSNorm epsilon (1e-6).
    pub connector_eps: f32,
    /// Acoustic latent noise std (fix_std = 0.5).
    pub acoustic_fix_std: f32,
    /// Acoustic sampling mode ("gaussian" adds `randn·(fix_std/0.8)` noise).
    pub acoustic_std_dist_type: String,
}

pub fn is_vibevoice_asr_mmproj(source: &dyn TensorSource) -> bool {
    source
        .metadata("general.architecture")
        .and_then(MetaValue::to_string_val)
        == Some("clip")
        && matches!(
            source.metadata("clip.has_vision_encoder"),
            Some(MetaValue::Bool(false))
        )
        && matches!(
            source.metadata("clip.has_audio_encoder"),
            Some(MetaValue::Bool(true))
        )
        && matches!(
            source.metadata("clip.has_gen_audio_encoder"),
            Some(MetaValue::Bool(false))
        )
        && source
            .metadata("clip.audio.projector_type")
            .and_then(MetaValue::to_string_val)
            == Some("vibevoice_asr")
}

fn meta_u64(source: &dyn TensorSource, key: &str) -> Result<u64, String> {
    match source.metadata(key) {
        Some(MetaValue::Uint64(value)) => Ok(*value),
        Some(MetaValue::Uint32(value)) => Ok(u64::from(*value)),
        _ => Err(format!("Invalid metadata: {key}; expected Uint64")),
    }
}

fn meta_f64(source: &dyn TensorSource, key: &str) -> Result<f64, String> {
    match source.metadata(key) {
        Some(MetaValue::Float64(value)) => Ok(*value),
        _ => Err(format!("Invalid metadata: {key}; expected Float64")),
    }
}

fn meta_string(source: &dyn TensorSource, key: &str) -> Result<String, String> {
    match source.metadata(key) {
        Some(MetaValue::String(value)) => Ok(value.clone()),
        _ => Err(format!("Invalid metadata: {key}; expected String")),
    }
}

fn meta_u64_array(source: &dyn TensorSource, key: &str) -> Result<Vec<usize>, String> {
    match source.metadata(key) {
        Some(MetaValue::Array(_, values)) => values
            .iter()
            .map(|value| match value {
                MetaValue::Uint64(v) => usize::try_from(*v)
                    .map_err(|_| format!("Invalid metadata: {key} element overflow")),
                MetaValue::Uint32(v) => usize::try_from(u64::from(*v))
                    .map_err(|_| format!("Invalid metadata: {key} element overflow")),
                _ => Err(format!("Invalid metadata: {key}; expected uint elements")),
            })
            .collect(),
        _ => Err(format!("Invalid metadata: {key}; expected uint array")),
    }
}

impl VibeVoiceAsrConfig {
    pub fn from_source(source: &dyn TensorSource) -> Result<Self, String> {
        if !is_vibevoice_asr_mmproj(source) {
            return Err(
                "mmproj is not a VibeVoice ASR frontend (missing vibevoice_asr projector metadata)"
                    .into(),
            );
        }
        let positive = |key: &str, value: u64| -> Result<usize, String> {
            usize::try_from(value)
                .ok()
                .filter(|v| *v > 0)
                .ok_or_else(|| format!("Invalid metadata: {key}; expected positive integer"))
        };
        let sample_rate = positive(
            "vibevoice.sample_rate",
            meta_u64(source, "vibevoice.sample_rate")?,
        )?;
        let compress_ratio = positive(
            "vibevoice.compress_ratio",
            meta_u64(source, "vibevoice.compress_ratio")?,
        )?;
        let chunk_frames = positive(
            "vibevoice.chunk_frames",
            meta_u64(source, "vibevoice.chunk_frames")?,
        )?;
        let lookahead_frames = positive(
            "vibevoice.lookahead_frames",
            meta_u64(source, "vibevoice.lookahead_frames")?,
        )?;
        let llm_hidden_size = positive(
            "vibevoice.llm_hidden_size",
            meta_u64(source, "vibevoice.llm_hidden_size")?,
        )?;
        let acoustic_vae_dim = positive(
            "vibevoice.acoustic.vae_dim",
            meta_u64(source, "vibevoice.acoustic.vae_dim")?,
        )?;
        let semantic_vae_dim = positive(
            "vibevoice.semantic.vae_dim",
            meta_u64(source, "vibevoice.semantic.vae_dim")?,
        )?;
        let n_filters = positive(
            "vibevoice.encoder.n_filters",
            meta_u64(source, "vibevoice.encoder.n_filters")?,
        )?;
        let ratios = meta_u64_array(source, "vibevoice.encoder.ratios")?;
        let depths = meta_u64_array(source, "vibevoice.encoder.depths")?;
        if ratios.is_empty() || depths.is_empty() {
            return Err("Invalid metadata: vibevoice.encoder ratios/depths are empty".into());
        }
        if depths.len() != ratios.len() + 1 {
            return Err(format!(
                "Invalid metadata: vibevoice.encoder.depths length {} must be ratios length {} + 1",
                depths.len(),
                ratios.len()
            ));
        }
        let kernel_size = positive(
            "vibevoice.encoder.kernel_size",
            meta_u64(source, "vibevoice.encoder.kernel_size")?,
        )?;
        let last_kernel_size = positive(
            "vibevoice.encoder.last_kernel_size",
            meta_u64(source, "vibevoice.encoder.last_kernel_size")?,
        )?;
        let ffn_expansion = positive(
            "vibevoice.encoder.ffn_expansion",
            meta_u64(source, "vibevoice.encoder.ffn_expansion")?,
        )?;
        let layernorm_eps = meta_f64(source, "vibevoice.encoder.layernorm_eps")?;
        let connector_eps = meta_f64(source, "vibevoice.connector.eps")?;
        let fix_std = meta_f64(source, "vibevoice.acoustic.fix_std")?;
        if !(layernorm_eps.is_finite() && layernorm_eps > 0.0)
            || !(connector_eps.is_finite() && connector_eps > 0.0)
            || !fix_std.is_finite()
            || fix_std < 0.0
        {
            return Err("Invalid metadata: vibevoice eps/fix_std out of range".into());
        }
        let pad_mode = meta_string(source, "vibevoice.encoder.pad_mode")?;
        if pad_mode != "constant" {
            return Err(format!(
                "Unsupported metadata: vibevoice.encoder.pad_mode {pad_mode:?}; only constant is supported"
            ));
        }
        let mixer_layer = meta_string(source, "vibevoice.encoder.mixer_layer")?;
        if mixer_layer != "depthwise_conv" {
            return Err(format!(
                "Unsupported metadata: vibevoice.encoder.mixer_layer {mixer_layer:?}"
            ));
        }
        let std_dist_type = meta_string(source, "vibevoice.acoustic.std_dist_type")?;
        if std_dist_type != "gaussian" && std_dist_type != "none" {
            return Err(format!(
                "Unsupported metadata: vibevoice.acoustic.std_dist_type {std_dist_type:?}"
            ));
        }
        Ok(Self {
            sample_rate,
            compress_ratio,
            chunk_frames,
            lookahead_frames,
            llm_hidden_size,
            acoustic_vae_dim,
            semantic_vae_dim,
            n_filters,
            ratios,
            depths,
            kernel_size,
            last_kernel_size,
            ffn_expansion,
            pad_mode,
            mixer_layer,
            layernorm_eps: layernorm_eps as f32,
            connector_eps: connector_eps as f32,
            acoustic_fix_std: fix_std as f32,
            acoustic_std_dist_type: std_dist_type,
        })
    }

    /// Runtime (reversed) downsample strides of the ConvNeXt encoder.
    pub fn encoder_strides(&self) -> Vec<usize> {
        self.ratios.iter().rev().copied().collect()
    }

    /// Latent width of the named tokenizer ("acoustic" or "semantic").
    pub fn vae_dim_for(&self, side: &str) -> Result<usize, String> {
        match side {
            "acoustic" => Ok(self.acoustic_vae_dim),
            "semantic" => Ok(self.semantic_vae_dim),
            other => Err(format!("unknown vibevoice tokenizer side {other:?}")),
        }
    }

    /// Latent frames emitted for exactly `samples` input audio frames.
    pub fn latent_frames(&self, samples: usize) -> usize {
        samples / self.compress_ratio
    }

    /// Samples per streaming encode window: (chunk + lookahead) frames.
    pub fn window_samples(&self) -> usize {
        (self.chunk_frames + self.lookahead_frames) * self.compress_ratio
    }

    /// Committed samples per streaming chunk (advance step between windows).
    pub fn chunk_samples(&self) -> usize {
        self.chunk_frames * self.compress_ratio
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::tensor::TensorInfo;
    use std::collections::HashMap;

    #[derive(Default)]
    struct Source {
        metadata: HashMap<String, MetaValue>,
    }

    impl TensorSource for Source {
        fn metadata(&self, key: &str) -> Option<&MetaValue> {
            self.metadata.get(key)
        }
        fn tensor_info(&self, _name: &str) -> Option<&TensorInfo> {
            None
        }
        fn tensor_slice(&self, _name: &str) -> Option<&[u8]> {
            None
        }
    }

    fn valid_metadata() -> HashMap<String, MetaValue> {
        let mut metadata = HashMap::from([
            (
                "general.architecture".into(),
                MetaValue::String("clip".into()),
            ),
            ("clip.has_vision_encoder".into(), MetaValue::Bool(false)),
            ("clip.has_audio_encoder".into(), MetaValue::Bool(true)),
            ("clip.has_gen_audio_encoder".into(), MetaValue::Bool(false)),
            (
                "clip.audio.projector_type".into(),
                MetaValue::String("vibevoice_asr".into()),
            ),
            ("vibevoice.sample_rate".into(), MetaValue::Uint64(24000)),
            ("vibevoice.compress_ratio".into(), MetaValue::Uint64(3200)),
            ("vibevoice.chunk_frames".into(), MetaValue::Uint64(22)),
            ("vibevoice.lookahead_frames".into(), MetaValue::Uint64(4)),
            ("vibevoice.llm_hidden_size".into(), MetaValue::Uint64(3584)),
            ("vibevoice.acoustic.vae_dim".into(), MetaValue::Uint64(64)),
            ("vibevoice.semantic.vae_dim".into(), MetaValue::Uint64(128)),
            ("vibevoice.encoder.n_filters".into(), MetaValue::Uint64(32)),
            ("vibevoice.encoder.kernel_size".into(), MetaValue::Uint64(7)),
            (
                "vibevoice.encoder.last_kernel_size".into(),
                MetaValue::Uint64(7),
            ),
            (
                "vibevoice.encoder.ffn_expansion".into(),
                MetaValue::Uint64(4),
            ),
            (
                "vibevoice.encoder.ratios".into(),
                MetaValue::Array(
                    MetaValueType::Uint32,
                    vec![
                        MetaValue::Uint32(8),
                        MetaValue::Uint32(5),
                        MetaValue::Uint32(5),
                        MetaValue::Uint32(4),
                        MetaValue::Uint32(2),
                        MetaValue::Uint32(2),
                    ],
                ),
            ),
            (
                "vibevoice.encoder.depths".into(),
                MetaValue::Array(
                    MetaValueType::Uint32,
                    vec![
                        MetaValue::Uint32(3),
                        MetaValue::Uint32(3),
                        MetaValue::Uint32(3),
                        MetaValue::Uint32(3),
                        MetaValue::Uint32(3),
                        MetaValue::Uint32(3),
                        MetaValue::Uint32(8),
                    ],
                ),
            ),
            (
                "vibevoice.encoder.pad_mode".into(),
                MetaValue::String("constant".into()),
            ),
            (
                "vibevoice.encoder.mixer_layer".into(),
                MetaValue::String("depthwise_conv".into()),
            ),
            (
                "vibevoice.encoder.layernorm_eps".into(),
                MetaValue::Float64(1e-5),
            ),
            ("vibevoice.connector.eps".into(), MetaValue::Float64(1e-6)),
            ("vibevoice.acoustic.fix_std".into(), MetaValue::Float64(0.5)),
            (
                "vibevoice.acoustic.std_dist_type".into(),
                MetaValue::String("gaussian".into()),
            ),
        ]);
        metadata
    }

    #[test]
    fn config_parses_streaming_metadata_and_derives_windows() {
        let source = Source {
            metadata: valid_metadata(),
        };
        let config = VibeVoiceAsrConfig::from_source(&source).unwrap();
        assert_eq!(config.sample_rate, 24000);
        assert_eq!(config.acoustic_vae_dim, 64);
        assert_eq!(config.encoder_strides(), vec![2, 2, 4, 5, 5, 8]);
        assert_eq!(config.depths, vec![3, 3, 3, 3, 3, 3, 8]);
        assert_eq!(config.window_samples(), 83200);
        assert_eq!(config.chunk_samples(), 70400);
        assert_eq!(config.latent_frames(83200), 26);
    }

    #[test]
    fn config_rejects_missing_projector_or_bad_depths() {
        let mut source = Source {
            metadata: valid_metadata(),
        };
        source.metadata.insert(
            "clip.audio.projector_type".into(),
            MetaValue::String("other".into()),
        );
        assert!(VibeVoiceAsrConfig::from_source(&source).is_err());

        let mut source = Source {
            metadata: valid_metadata(),
        };
        source.metadata.insert(
            "vibevoice.encoder.depths".into(),
            MetaValue::Array(MetaValueType::Uint32, vec![MetaValue::Uint32(3)]),
        );
        assert!(VibeVoiceAsrConfig::from_source(&source).is_err());
    }
}
