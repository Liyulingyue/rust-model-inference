//! Sophon-Realtime component contracts. The thinker uses the unchanged Qwen3-VL decoder.

fn check_cancelled(cancelled: &dyn Fn() -> bool) -> Result<(), String> {
    if cancelled() {
        Err("speech synthesis cancelled".into())
    } else {
        Ok(())
    }
}

use crate::core::tensor::{require_tensor_meta, GGMLType, MetaValue, TensorInfo, TensorSource};
use std::sync::Arc;

pub const ARCH: &str = "sophon_realtime";
pub const SOURCE_REVISION: &str = "990d6a74bbf15c1765a6f7736fa839f423fd5b59";

pub mod audio;
pub mod flow;
pub mod hift;
pub mod speech;
pub mod talker;

pub fn is_thinker(source: &dyn TensorSource) -> bool {
    source
        .metadata("sophon_realtime.component")
        .and_then(MetaValue::to_string_val)
        == Some("thinker")
}

fn require_string(source: &dyn TensorSource, key: &str, expected: &str) -> Result<(), String> {
    if source.metadata(key).and_then(MetaValue::to_string_val) != Some(expected) {
        return Err(format!("Sophon requires {key}={expected}"));
    }
    Ok(())
}

fn require_u64(source: &dyn TensorSource, key: &str, expected: u64) -> Result<(), String> {
    if source.metadata(key).and_then(MetaValue::to_u64) != Some(expected) {
        return Err(format!("Sophon requires {key}={expected}"));
    }
    Ok(())
}

/// Only the pinned, fully checked thinker is equivalent to this decoder.
pub fn decoder_source(source: Arc<dyn TensorSource>) -> Result<Arc<dyn TensorSource>, String> {
    if source
        .metadata("general.architecture")
        .and_then(MetaValue::to_string_val)
        != Some(ARCH)
    {
        return Ok(source);
    }
    require_string(source.as_ref(), "sophon_realtime.component", "thinker")?;
    require_string(
        source.as_ref(),
        "sophon_realtime.source_revision",
        SOURCE_REVISION,
    )?;
    for (key, value) in [
        ("embedding_length", 5120),
        ("block_count", 64),
        ("attention.head_count", 64),
        ("attention.head_count_kv", 8),
        ("attention.key_length", 128),
        ("attention.value_length", 128),
        ("feed_forward_length", 25600),
        ("n_deepstack_layers", 3),
        ("context_length", 262144),
    ] {
        require_u64(source.as_ref(), &format!("qwen3vl.{key}"), value)?;
    }
    for (key, value) in [
        ("qwen3vl.rope.freq_base", 5_000_000.0),
        ("qwen3vl.attention.layer_norm_rms_epsilon", 1e-6),
    ] {
        if source
            .metadata(key)
            .and_then(MetaValue::to_f64)
            .map(|value| value as f32)
            != Some(value)
        {
            return Err(format!("Sophon requires {key}={value}"));
        }
    }
    let sections = source
        .metadata("qwen3vl.rope.dimension_sections")
        .and_then(MetaValue::to_arr)
        .and_then(|values| {
            values
                .iter()
                .map(MetaValue::to_u64)
                .collect::<Option<Vec<_>>>()
        });
    if sections.as_deref() != Some(&[24, 20, 20, 0]) {
        return Err("Sophon requires interleaved rope.dimension_sections=[24,20,20,0]".into());
    }
    let matrices = &[
        GGMLType::F32,
        GGMLType::F16,
        GGMLType::BF16,
        GGMLType::Q8_0,
        GGMLType::Q4_0,
        GGMLType::Q4_1,
        GGMLType::Q4K,
        GGMLType::Q5K,
        GGMLType::Q6K,
    ];
    for name in ["token_embd.weight", "output.weight"] {
        require_tensor_meta(source.as_ref(), name, &[5120, 151936], matrices)?;
    }
    require_tensor_meta(
        source.as_ref(),
        "output_norm.weight",
        &[5120],
        &[GGMLType::F32],
    )?;
    for layer in 0..64 {
        for (name, width) in [
            ("attn_norm", 5120),
            ("ffn_norm", 5120),
            ("attn_q_norm", 128),
            ("attn_k_norm", 128),
        ] {
            require_tensor_meta(
                source.as_ref(),
                &format!("blk.{layer}.{name}.weight"),
                &[width],
                &[GGMLType::F32],
            )?;
        }
        for (name, dims) in [
            ("attn_q", [5120, 8192]),
            ("attn_k", [5120, 1024]),
            ("attn_v", [5120, 1024]),
            ("attn_output", [8192, 5120]),
            ("ffn_gate", [5120, 25600]),
            ("ffn_up", [5120, 25600]),
            ("ffn_down", [25600, 5120]),
        ] {
            require_tensor_meta(
                source.as_ref(),
                &format!("blk.{layer}.{name}.weight"),
                &dims,
                matrices,
            )?;
        }
    }
    // Transformers discards the whitespace after this release's final Jinja block.
    let template = source
        .metadata("tokenizer.chat_template")
        .and_then(MetaValue::to_string_val)
        .map(|raw| MetaValue::String(raw.trim_end_matches(['\r', '\n']).to_owned()));
    Ok(Arc::new(ThinkerSource {
        source,
        architecture: MetaValue::String("qwen3vl".into()),
        template,
    }))
}

/// Check pairing before the generic vision loader can accept an unrelated tower.
pub fn validate_vision(source: &dyn TensorSource) -> Result<(), String> {
    require_string(source, "general.architecture", "clip")?;
    require_string(source, "sophon_realtime.component", "vision")?;
    require_string(source, "sophon_realtime.source_revision", SOURCE_REVISION)?;
    require_string(source, "clip.projector_type", "qwen3vl_merger")?;
    for (key, value) in [
        ("embedding_length", 1152),
        ("feed_forward_length", 4304),
        ("block_count", 27),
        ("attention.head_count", 16),
        ("patch_size", 16),
        ("spatial_merge_size", 2),
        ("projection_dim", 5120),
    ] {
        require_u64(source, &format!("clip.vision.{key}"), value)?;
    }
    let layers = source
        .metadata("clip.vision.is_deepstack_layers")
        .and_then(MetaValue::to_arr);
    if !matches!(layers, Some(values) if values.len() == 27 && values.iter().enumerate().all(|(i, value)| matches!(value, MetaValue::Bool(flag) if *flag == [8,16,24].contains(&i))))
    {
        return Err("Sophon vision requires DeepStack layers [8,16,24]".into());
    }
    Ok(())
}

struct ThinkerSource {
    source: Arc<dyn TensorSource>,
    architecture: MetaValue,
    template: Option<MetaValue>,
}

impl TensorSource for ThinkerSource {
    fn borrowed_static_f32_weights(&self) -> bool {
        // Qwen3Model retains this adapter's Arc alongside all static weights.
        true
    }
    fn metadata(&self, key: &str) -> Option<&MetaValue> {
        if key == "general.architecture" {
            Some(&self.architecture)
        } else if key == "tokenizer.chat_template" {
            self.template.as_ref()
        } else {
            self.source.metadata(key)
        }
    }
    fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.source.tensor_info(name)
    }
    fn tensor_slice(&self, name: &str) -> Option<&[u8]> {
        self.source.tensor_slice(name)
    }
}

#[cfg(test)]
mod tests;
