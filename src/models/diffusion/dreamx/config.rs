use crate::core::tensor::{MetaValue, TensorSource};

const COMPONENTS: [&str; 10] = [
    "audio_vae",
    "creator.audio",
    "creator.joint",
    "creator.video",
    "refiner.dit",
    "refiner.lightvae",
    "refiner.upsampler.causal2d",
    "refiner.upsampler.flash",
    "text",
    "video_vae",
];
const MAIN_COMPONENTS: [&str; 4] = [
    "creator.audio",
    "creator.joint",
    "creator.video",
    "refiner.dit",
];
const MMPROJ_COMPONENTS: [&str; 6] = [
    "audio_vae",
    "refiner.lightvae",
    "refiner.upsampler.causal2d",
    "refiner.upsampler.flash",
    "text",
    "video_vae",
];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DreamXConfig {
    pub pair_id: String,
    pub video_embedding_length: usize,
    pub video_feed_forward_length: usize,
    pub video_head_count: usize,
    pub video_block_count: usize,
    pub video_in_channels: usize,
    pub audio_embedding_length: usize,
    pub audio_feed_forward_length: usize,
    pub audio_head_count: usize,
    pub audio_block_count: usize,
    pub audio_in_channels: usize,
    pub text_context_length: usize,
    pub text_embedding_length: usize,
    pub text_feed_forward_length: usize,
    pub text_head_count: usize,
    pub text_block_count: usize,
    pub text_vocab_size: usize,
}

impl DreamXConfig {
    pub fn from_sources(
        main: &dyn TensorSource,
        mmproj: &dyn TensorSource,
    ) -> Result<Self, String> {
        require_string(main, "general.architecture", "dreamx")?;
        require_string(mmproj, "general.architecture", "clip")?;
        require_string(mmproj, "general.type", "mmproj")?;
        require_string(mmproj, "clip.projector_type", "dreamx_creator")?;

        let pair_id = metadata_string(main, "dreamx.pair_id")?;
        let mmproj_pair_id = metadata_string(mmproj, "dreamx.pair_id")?;
        if pair_id != mmproj_pair_id {
            return Err(format!(
                "DreamX pair ID mismatch: main {pair_id:?}, mmproj {mmproj_pair_id:?}"
            ));
        }
        require_sha256("dreamx.pair_id", &pair_id)?;

        require_string(main, "dreamx.file_role", "main")?;
        require_string(mmproj, "dreamx.file_role", "mmproj")?;
        require_string(main, "dreamx.source_model", "GD-ML/DreamX-Creator")?;
        require_string(mmproj, "dreamx.source_model", "GD-ML/DreamX-Creator")?;
        require_u64(main, "dreamx.exporter_version", 1)?;
        require_u64(mmproj, "dreamx.exporter_version", 1)?;
        require_string_array(main, "dreamx.components", &COMPONENTS)?;
        require_string_array(mmproj, "dreamx.components", &COMPONENTS)?;
        require_string_array(main, "dreamx.file_components", &MAIN_COMPONENTS)?;
        require_string_array(mmproj, "dreamx.file_components", &MMPROJ_COMPONENTS)?;
        require_u64_array(main, "dreamx.joint_layers", &(15..=29).collect::<Vec<_>>())?;
        require_u64_array(
            mmproj,
            "dreamx.joint_layers",
            &(15..=29).collect::<Vec<_>>(),
        )?;

        let tokenizer_sha = metadata_string(main, "dreamx.tokenizer.sha256")?;
        if tokenizer_sha != metadata_string(mmproj, "dreamx.tokenizer.sha256")? {
            return Err("DreamX tokenizer SHA-256 mismatch".into());
        }
        require_sha256("dreamx.tokenizer.sha256", &tokenizer_sha)?;

        for component in COMPONENTS {
            let flag = format!("dreamx.has_component.{component}");
            require_bool(main, &flag, true)?;
            require_bool(mmproj, &flag, true)?;
            let count_key = format!("dreamx.component.{component}.tensor_count");
            let main_count = metadata_u64(main, &count_key)?;
            let mmproj_count = metadata_u64(mmproj, &count_key)?;
            if main_count == 0 || main_count != mmproj_count {
                return Err(format!(
                    "DreamX component inventory mismatch for {component}: main {main_count}, mmproj {mmproj_count}"
                ));
            }
        }

        for (key, expected) in RELEASED_DIMENSIONS {
            require_u64(main, key, expected)?;
            require_u64(mmproj, key, expected)?;
        }

        Ok(Self {
            pair_id,
            video_embedding_length: 3072,
            video_feed_forward_length: 14336,
            video_head_count: 24,
            video_block_count: 30,
            video_in_channels: 48,
            audio_embedding_length: 1536,
            audio_feed_forward_length: 8960,
            audio_head_count: 12,
            audio_block_count: 30,
            audio_in_channels: 128,
            text_context_length: 512,
            text_embedding_length: 4096,
            text_feed_forward_length: 10240,
            text_head_count: 64,
            text_block_count: 24,
            text_vocab_size: 256384,
        })
    }
}

const RELEASED_DIMENSIONS: [(&str, u64); 16] = [
    ("dreamx.video.embedding_length", 3072),
    ("dreamx.video.feed_forward_length", 14336),
    ("dreamx.video.attention.head_count", 24),
    ("dreamx.video.block_count", 30),
    ("dreamx.video.in_channels", 48),
    ("dreamx.audio.embedding_length", 1536),
    ("dreamx.audio.feed_forward_length", 8960),
    ("dreamx.audio.attention.head_count", 12),
    ("dreamx.audio.block_count", 30),
    ("dreamx.audio.in_channels", 128),
    ("dreamx.text.context_length", 512),
    ("dreamx.text.embedding_length", 4096),
    ("dreamx.text.feed_forward_length", 10240),
    ("dreamx.text.attention.head_count", 64),
    ("dreamx.text.block_count", 24),
    ("dreamx.text.vocab_size", 256384),
];

fn metadata_string(source: &dyn TensorSource, key: &str) -> Result<String, String> {
    source
        .metadata(key)
        .and_then(MetaValue::to_string_val)
        .map(str::to_owned)
        .ok_or_else(|| format!("Invalid DreamX metadata {key}: expected string"))
}

fn metadata_u64(source: &dyn TensorSource, key: &str) -> Result<u64, String> {
    source
        .metadata(key)
        .and_then(MetaValue::to_u64)
        .ok_or_else(|| format!("Invalid DreamX metadata {key}: expected integer"))
}

fn require_string(source: &dyn TensorSource, key: &str, expected: &str) -> Result<(), String> {
    let actual = metadata_string(source, key)?;
    if actual == expected {
        Ok(())
    } else {
        Err(format!(
            "Invalid DreamX metadata {key}: expected {expected:?}, got {actual:?}"
        ))
    }
}

fn require_u64(source: &dyn TensorSource, key: &str, expected: u64) -> Result<(), String> {
    let actual = metadata_u64(source, key)?;
    if actual == expected {
        Ok(())
    } else {
        Err(format!(
            "Invalid DreamX metadata {key}: expected {expected}, got {actual}"
        ))
    }
}

fn require_bool(source: &dyn TensorSource, key: &str, expected: bool) -> Result<(), String> {
    match source.metadata(key) {
        Some(MetaValue::Bool(actual)) if *actual == expected => Ok(()),
        Some(MetaValue::Bool(actual)) => Err(format!(
            "Invalid DreamX metadata {key}: expected {expected}, got {actual}"
        )),
        _ => Err(format!("Invalid DreamX metadata {key}: expected boolean")),
    }
}

fn metadata_array<'a>(source: &'a dyn TensorSource, key: &str) -> Result<&'a [MetaValue], String> {
    match source.metadata(key) {
        Some(MetaValue::Array(_, values)) => Ok(values),
        _ => Err(format!("Invalid DreamX metadata {key}: expected array")),
    }
}

fn require_string_array(
    source: &dyn TensorSource,
    key: &str,
    expected: &[&str],
) -> Result<(), String> {
    let actual = metadata_array(source, key)?
        .iter()
        .map(|value| {
            value
                .to_string_val()
                .ok_or_else(|| format!("Invalid DreamX metadata {key}: expected string array"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if actual == expected {
        Ok(())
    } else {
        Err(format!("DreamX component inventory mismatch in {key}"))
    }
}

fn require_u64_array(source: &dyn TensorSource, key: &str, expected: &[u64]) -> Result<(), String> {
    let actual = metadata_array(source, key)?
        .iter()
        .map(|value| {
            value
                .to_u64()
                .ok_or_else(|| format!("Invalid DreamX metadata {key}: expected integer array"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if actual == expected {
        Ok(())
    } else {
        Err(format!(
            "Invalid DreamX metadata {key}: expected released joint layers"
        ))
    }
}

fn require_sha256(key: &str, value: &str) -> Result<(), String> {
    if value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        Ok(())
    } else {
        Err(format!(
            "Invalid DreamX metadata {key}: expected 64 hexadecimal characters"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::tensor::{MetaValue, TensorInfo, TensorSource};
    use std::collections::HashMap;

    #[derive(Default)]
    struct Source {
        metadata: HashMap<String, MetaValue>,
    }

    impl Source {
        fn with(mut self, key: &str, value: MetaValue) -> Self {
            self.metadata.insert(key.into(), value);
            self
        }
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

    fn source(architecture: &str, pair_id: &str) -> Source {
        Source::default()
            .with(
                "general.architecture",
                MetaValue::String(architecture.into()),
            )
            .with("dreamx.pair_id", MetaValue::String(pair_id.into()))
    }

    #[test]
    fn pair_validation_rejects_different_ids() {
        let main = source("dreamx", &"a".repeat(64));
        let aux = source("clip", &"b".repeat(64))
            .with("general.type", MetaValue::String("mmproj".into()))
            .with(
                "clip.projector_type",
                MetaValue::String("dreamx_creator".into()),
            );
        assert!(DreamXConfig::from_sources(&main, &aux)
            .unwrap_err()
            .contains("pair ID mismatch"));
    }
}
