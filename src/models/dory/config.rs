//! GGUF contract for NVIDIA Nemotron-Labs Dory 1.0.

use crate::core::tensor::{MetaValue, TensorSource};

#[derive(Debug, Clone)]
pub struct DoryConfig {
    pub n_embd: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub n_ff: usize,
    pub n_ctx: usize,
    pub vocab_size: usize,
    pub rope_freq_base: f32,
    pub rope_freq_base_2: f32,
    pub sliding_window: usize,
    pub n_input_layers: usize,
    pub n_recurrent_layers: usize,
    pub n_output_layers: usize,
    pub n_loops: usize,
    pub recurrent_kv_cache_mode: String,
    pub initializer_range: f32,
    pub alpha_init: f32,
    pub gate_init: f32,
    pub rope_profile: Vec<usize>,
    /// `true` for attention layers using the short (1e4) RoPE profile.
    pub attention_sliding: Vec<bool>,
}

fn u(source: &dyn TensorSource, key: &str) -> Result<usize, String> {
    source
        .metadata(key)
        .and_then(MetaValue::to_u64)
        .and_then(|v| usize::try_from(v).ok())
        .ok_or_else(|| format!("Missing or invalid metadata: {key}"))
}

fn f(source: &dyn TensorSource, key: &str) -> Result<f32, String> {
    let value = source
        .metadata(key)
        .and_then(MetaValue::to_f64)
        .map(|v| v as f32)
        .ok_or_else(|| format!("Missing or invalid metadata: {key}"))?;
    if !value.is_finite() || value <= 0.0 {
        return Err(format!("Invalid positive metadata {key}: {value}"));
    }
    Ok(value)
}

fn string(source: &dyn TensorSource, key: &str) -> Result<String, String> {
    source
        .metadata(key)
        .and_then(MetaValue::to_string_val)
        .map(ToOwned::to_owned)
        .ok_or_else(|| format!("Missing or invalid metadata: {key}"))
}

fn uint_array(source: &dyn TensorSource, key: &str, len: usize) -> Result<Vec<usize>, String> {
    let value = source
        .metadata(key)
        .ok_or_else(|| format!("Missing metadata: {key}"))?;
    let MetaValue::Array(_, values) = value else {
        return Err(format!("{key} must be an array"));
    };
    if values.len() != len {
        return Err(format!(
            "{key} expected {len} entries, got {}",
            values.len()
        ));
    }
    values
        .iter()
        .map(|value| {
            value
                .to_u64()
                .and_then(|v| usize::try_from(v).ok())
                .ok_or_else(|| format!("{key} has a non-integer entry"))
        })
        .collect()
}

impl DoryConfig {
    pub fn from_source(source: &dyn TensorSource) -> Result<Self, String> {
        let arch = source
            .metadata("general.architecture")
            .and_then(MetaValue::to_string_val)
            .ok_or_else(|| "Missing general.architecture".to_string())?;
        if arch != "dory" {
            return Err(format!("Unsupported architecture for DoryConfig: {arch}"));
        }
        let n_layer = u(source, "dory.block_count")?;
        let n_input_layers = u(source, "dory.input_layer_count")?;
        let n_recurrent_layers = u(source, "dory.recurrent_layer_count")?;
        let n_output_layers = u(source, "dory.output_layer_count")?;
        if n_input_layers
            .checked_add(n_recurrent_layers)
            .and_then(|v| v.checked_add(n_output_layers))
            != Some(n_layer)
        {
            return Err(format!(
                "Dory layer partition does not cover block_count: {n_input_layers}+{n_recurrent_layers}+{n_output_layers} != {n_layer}"
            ));
        }
        if n_input_layers % 2 != 0 || n_recurrent_layers % 2 != 0 || n_output_layers % 2 != 0 {
            return Err("Dory layer partitions must contain attention/MLP pairs".into());
        }
        if n_input_layers == 0
            || n_recurrent_layers == 0
            || n_output_layers == 0
            || !(1..=4).contains(&u(source, "dory.num_loops")?)
        {
            return Err(
                "Dory layer partitions and num_loops must be non-zero (loops 1..=4)".into(),
            );
        }
        let pattern = string(source, "dory.layer_pattern")?;
        if pattern.chars().count() != n_layer
            || pattern.chars().enumerate().any(|(index, value)| {
                if index % 2 == 0 {
                    value != 'S' && value != '*'
                } else {
                    value != '-'
                }
            })
        {
            return Err("dory.layer_pattern must alternate attention and MLP entries".into());
        }
        let attention_layers = n_layer / 2;
        let mut attention_sliding = Vec::with_capacity(attention_layers);
        for item in pattern.trim_end_matches('-').split('-') {
            match item {
                "S" => attention_sliding.push(true),
                "*" => attention_sliding.push(false),
                other => return Err(format!("Invalid dory.layer_pattern item {other:?}")),
            }
        }
        if attention_sliding.len() != attention_layers {
            return Err(format!(
                "dory.layer_pattern expected {attention_layers} attention entries, got {}",
                attention_sliding.len()
            ));
        }
        let rope_profile = uint_array(source, "dory.rope.profile", n_layer)?;
        if rope_profile.iter().any(|&value| value != 1 && value != 2) {
            return Err("dory.rope.profile entries must be 1 or 2".into());
        }
        for (attention_index, &sliding) in attention_sliding.iter().enumerate() {
            let profile = rope_profile[attention_index * 2];
            let expected = if sliding { 2 } else { 1 };
            if profile != expected {
                return Err(format!(
                    "dory.rope.profile disagrees with layer_pattern at attention layer {}: expected {expected}, got {profile}",
                    attention_index * 2
                ));
            }
        }
        let cache_mode = string(source, "dory.recurrent_kv_cache_mode")?;
        if cache_mode != "per_loop" && cache_mode != "last_loop" {
            return Err(format!(
                "Unsupported Dory recurrent KV cache mode: {cache_mode}"
            ));
        }
        let cfg = Self {
            n_embd: u(source, "dory.embedding_length")?,
            n_layer,
            n_head: u(source, "dory.attention.head_count")?,
            n_head_kv: u(source, "dory.attention.head_count_kv")?,
            head_dim: u(source, "dory.attention.key_length")?,
            n_ff: u(source, "dory.feed_forward_length")?,
            n_ctx: u(source, "dory.context_length")?,
            vocab_size: source
                .metadata("tokenizer.ggml.tokens")
                .and_then(MetaValue::to_arr)
                .map(Vec::len)
                .ok_or_else(|| "Missing tokenizer.ggml.tokens".to_string())?,
            rope_freq_base: f(source, "dory.rope.freq_base")?,
            rope_freq_base_2: f(source, "dory.rope.freq_base_2")?,
            sliding_window: u(source, "dory.attention.sliding_window")?,
            n_input_layers,
            n_recurrent_layers,
            n_output_layers,
            n_loops: u(source, "dory.num_loops")?,
            recurrent_kv_cache_mode: cache_mode,
            initializer_range: f(source, "dory.init_std")?,
            alpha_init: f(source, "dory.alpha_init")?,
            gate_init: f(source, "dory.gate_init")?,
            rope_profile,
            attention_sliding,
        };
        if cfg.n_embd == 0
            || cfg.n_ff == 0
            || cfg.n_ctx == 0
            || cfg.vocab_size == 0
            || cfg.n_head == 0
            || cfg.n_head_kv == 0
            || cfg.n_head % cfg.n_head_kv != 0
            || cfg.n_loops == 0
            || cfg.sliding_window == 0
            || u(source, "dory.rope.dimension_count")? != cfg.head_dim
            || u(source, "dory.attention.value_length")? != cfg.head_dim
        {
            return Err("Invalid Dory attention, loop, or sliding-window dimensions".into());
        }
        Ok(cfg)
    }

    pub fn is_attention(&self, layer: usize) -> bool {
        layer % 2 == 0
    }

    pub fn attention_is_sliding(&self, layer: usize) -> bool {
        debug_assert!(self.is_attention(layer));
        self.attention_sliding[layer / 2]
    }

    pub fn in_recurrent_block(&self, layer: usize) -> bool {
        (self.n_input_layers..self.n_input_layers + self.n_recurrent_layers).contains(&layer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::tensor::{MetaValueType, TensorInfo};
    use std::collections::HashMap;

    #[derive(Default)]
    struct Source {
        metadata: HashMap<String, MetaValue>,
    }

    impl TensorSource for Source {
        fn metadata(&self, key: &str) -> Option<&MetaValue> {
            self.metadata.get(key)
        }

        fn tensor_info(&self, _: &str) -> Option<&TensorInfo> {
            None
        }

        fn tensor_slice(&self, _: &str) -> Option<&[u8]> {
            None
        }
    }

    fn source() -> Source {
        let mut metadata = HashMap::new();
        metadata.insert(
            "general.architecture".into(),
            MetaValue::String("dory".into()),
        );
        for (key, value) in [
            ("dory.block_count", 6),
            ("dory.input_layer_count", 2),
            ("dory.recurrent_layer_count", 2),
            ("dory.output_layer_count", 2),
            ("dory.embedding_length", 8),
            ("dory.attention.head_count", 2),
            ("dory.attention.head_count_kv", 1),
            ("dory.attention.key_length", 4),
            ("dory.attention.value_length", 4),
            ("dory.feed_forward_length", 16),
            ("dory.context_length", 64),
            ("dory.attention.sliding_window", 8),
            ("dory.num_loops", 2),
            ("dory.rope.dimension_count", 4),
        ] {
            metadata.insert(key.into(), MetaValue::Uint64(value));
        }
        for (key, value) in [
            ("dory.rope.freq_base", 10_000.0),
            ("dory.rope.freq_base_2", 1_000.0),
            ("dory.init_std", 0.0177),
            ("dory.alpha_init", 0.05),
            ("dory.gate_init", 2.0),
        ] {
            metadata.insert(key.into(), MetaValue::Float64(value));
        }
        metadata.insert(
            "dory.layer_pattern".into(),
            MetaValue::String("S-*-S-".into()),
        );
        metadata.insert(
            "dory.recurrent_kv_cache_mode".into(),
            MetaValue::String("per_loop".into()),
        );
        metadata.insert(
            "dory.rope.profile".into(),
            MetaValue::Array(
                MetaValueType::Uint64,
                [2, 1, 1, 1, 2, 1]
                    .into_iter()
                    .map(MetaValue::Uint64)
                    .collect(),
            ),
        );
        metadata.insert(
            "tokenizer.ggml.tokens".into(),
            MetaValue::Array(
                MetaValueType::String,
                vec![MetaValue::String("<s>".into()); 32],
            ),
        );
        Source { metadata }
    }

    #[test]
    fn rejects_wrong_architecture() {
        let mut source = source();
        source.metadata.insert(
            "general.architecture".into(),
            MetaValue::String("llama".into()),
        );
        assert!(DoryConfig::from_source(&source)
            .unwrap_err()
            .contains("Unsupported architecture"));
    }

    #[test]
    fn rejects_partition_that_does_not_cover_blocks() {
        let mut source = source();
        source
            .metadata
            .insert("dory.output_layer_count".into(), MetaValue::Uint64(4));
        assert!(DoryConfig::from_source(&source)
            .unwrap_err()
            .contains("does not cover block_count"));
    }

    #[test]
    fn rejects_invalid_layer_pattern() {
        let mut source = source();
        source.metadata.insert(
            "dory.layer_pattern".into(),
            MetaValue::String("S-X-S-".into()),
        );
        assert!(DoryConfig::from_source(&source)
            .unwrap_err()
            .contains("layer_pattern"));
    }

    #[test]
    fn accepts_complete_contract() {
        let config = DoryConfig::from_source(&source()).unwrap();
        assert_eq!(config.n_layer, 6);
        assert_eq!(config.attention_sliding, [true, false, true]);
        assert_eq!(config.rope_profile, [2, 1, 1, 1, 2, 1]);
        assert_eq!(config.vocab_size, 32);
    }
}
