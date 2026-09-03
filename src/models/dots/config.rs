//! dots.tts model configuration, derived from the mmproj GGUF metadata.

use crate::core::tensor::{MetaValue, TensorSource};

#[derive(Debug, Clone)]
pub struct DotsTtsConfig {
    /// Number of latent frames per patch (config.patch_size).
    pub patch_size: usize,
    /// AudioVAE latent width (128).
    pub latent_dim: usize,
    /// Latent frames per second of audio (48 kHz / hop 1920 = 25).
    pub hop_size: usize,
    /// Output sample rate (48000).
    pub sample_rate: usize,
    /// Flow-matching hidden width (DiT hidden_size, 1024).
    pub fm_hidden_size: usize,
    /// LLM hidden width (1536).
    pub llm_hidden_size: usize,
    /// Speaker x-vector width (512).
    pub xvec_dim: usize,
    /// Patch encoder transformer depth.
    pub patch_encoder_layers: usize,
    /// DiT block depth.
    pub dit_layers: usize,
    /// DiT attention heads.
    pub dit_heads: usize,
    /// Default Euler steps from the mmproj sampling metadata.
    pub default_nfe: usize,
    /// Default classifier-free guidance from the mmproj sampling metadata.
    pub default_guidance: f32,
    /// Default prompt speaker scale from the mmproj sampling metadata.
    pub default_speaker_scale: f32,
    /// Default EOS probability threshold from the mmproj sampling metadata.
    pub default_eos_threshold: f32,
}

pub fn is_dots_tts_mmproj(source: &dyn TensorSource) -> bool {
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
            Some(MetaValue::Bool(true))
        )
        && source
            .metadata("clip.audio.projector_type")
            .and_then(MetaValue::to_string_val)
            == Some("dotstts_spkenc")
        && source
            .metadata("clip.gen.audio.projector_type")
            .and_then(MetaValue::to_string_val)
            == Some("dotstts_gen")
}

fn clip_contract_error(source: &dyn TensorSource) -> String {
    for (key, expected) in [
        ("general.architecture", "clip"),
        ("clip.audio.projector_type", "dotstts_spkenc"),
        ("clip.gen.audio.projector_type", "dotstts_gen"),
    ] {
        if source.metadata(key).and_then(MetaValue::to_string_val) != Some(expected) {
            return format!("Missing or mismatched clip metadata: {key}");
        }
    }
    for (key, expected) in [
        ("clip.has_vision_encoder", false),
        ("clip.has_audio_encoder", true),
        ("clip.has_gen_audio_encoder", true),
    ] {
        if !matches!(source.metadata(key), Some(MetaValue::Bool(value)) if *value == expected) {
            return format!("Missing or mismatched clip metadata: {key}");
        }
    }
    "Invalid dots.tts clip projector metadata".into()
}

impl DotsTtsConfig {
    pub fn from_source(source: &dyn TensorSource) -> Result<Self, String> {
        if !is_dots_tts_mmproj(source) {
            return Err(clip_contract_error(source));
        }
        let u = |key: &str| -> Result<usize, String> {
            source
                .metadata(key)
                .and_then(MetaValue::to_u64)
                .map(|value| value as usize)
                .ok_or_else(|| format!("Missing metadata: {key}"))
        };
        let f = |key: &str| -> Result<f32, String> {
            source
                .metadata(key)
                .and_then(MetaValue::to_f64)
                .map(|value| value as f32)
                .ok_or_else(|| format!("Missing metadata: {key}"))
        };
        Ok(Self {
            patch_size: u("dotstts.patch_size")?,
            latent_dim: u("dotstts.latent_dim")?,
            hop_size: u("dotstts.hop_size")?,
            sample_rate: u("dotstts.sample_rate")?,
            fm_hidden_size: u("dotstts.fm_hidden_size")?,
            llm_hidden_size: u("dotstts.llm_hidden_size")?,
            xvec_dim: u("dotstts.xvec_dim")?,
            patch_encoder_layers: 24,
            dit_layers: 18,
            dit_heads: 16,
            default_nfe: u("dotstts.sampling.nfe")?,
            default_guidance: f("dotstts.sampling.guidance")?,
            default_speaker_scale: f("dotstts.sampling.speaker_scale")?,
            default_eos_threshold: f("dotstts.sampling.eos_threshold")?,
        })
    }

    /// Audio samples per patch (4 patches × 1920 frames = 7680 samples).
    pub fn samples_per_patch(&self) -> usize {
        self.patch_size * self.hop_size
    }

    /// Total FM sequence rows per audio patch: 1 hidden row + patch_size latent rows.
    pub fn unit_len(&self) -> usize {
        1 + self.patch_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::tensor::TensorInfo;

    #[derive(Default)]
    struct Source {
        metadata: std::collections::HashMap<String, MetaValue>,
        infos: std::collections::HashMap<String, TensorInfo>,
        bytes: std::collections::HashMap<String, Vec<u8>>,
    }

    impl TensorSource for Source {
        fn metadata(&self, key: &str) -> Option<&MetaValue> {
            self.metadata.get(key)
        }

        fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
            self.infos.get(name)
        }

        fn tensor_slice(&self, name: &str) -> Option<&[u8]> {
            self.bytes.get(name).map(Vec::as_slice)
        }
    }

    fn dots_metadata_source() -> Source {
        let mut metadata = std::collections::HashMap::from([
            (
                "general.architecture".into(),
                MetaValue::String("clip".into()),
            ),
            ("clip.has_vision_encoder".into(), MetaValue::Bool(false)),
            ("clip.has_audio_encoder".into(), MetaValue::Bool(true)),
            ("clip.has_gen_audio_encoder".into(), MetaValue::Bool(true)),
            (
                "clip.audio.projector_type".into(),
                MetaValue::String("dotstts_spkenc".into()),
            ),
            (
                "clip.gen.audio.projector_type".into(),
                MetaValue::String("dotstts_gen".into()),
            ),
        ]);
        for (key, value) in [
            ("dotstts.patch_size", 4),
            ("dotstts.latent_dim", 128),
            ("dotstts.hop_size", 1920),
            ("dotstts.sample_rate", 48_000),
            ("dotstts.fm_hidden_size", 1024),
            ("dotstts.llm_hidden_size", 1536),
            ("dotstts.xvec_dim", 512),
            ("dotstts.sampling.nfe", 10),
        ] {
            metadata.insert(key.into(), MetaValue::Uint64(value));
        }
        for (key, value) in [
            ("dotstts.sampling.guidance", 1.2),
            ("dotstts.sampling.speaker_scale", 1.5),
            ("dotstts.sampling.eos_threshold", 0.8),
        ] {
            metadata.insert(key.into(), MetaValue::Float64(value));
        }
        Source {
            metadata,
            ..Source::default()
        }
    }

    #[test]
    fn clip_projector_pair_identifies_dots_and_rejects_partial_matches() {
        let mut source = dots_metadata_source();
        assert!(is_dots_tts_mmproj(&source));
        let config = DotsTtsConfig::from_source(&source).unwrap();
        assert_eq!(config.patch_size, 4);
        assert_eq!(config.default_nfe, 10);
        assert_eq!(config.default_guidance, 1.2);
        assert_eq!(config.default_speaker_scale, 1.5);
        assert_eq!(config.default_eos_threshold, 0.8);
        source.metadata.insert(
            "clip.gen.audio.projector_type".into(),
            MetaValue::String("other".into()),
        );
        assert!(!is_dots_tts_mmproj(&source));
        assert!(DotsTtsConfig::from_source(&source).is_err());
    }
}
