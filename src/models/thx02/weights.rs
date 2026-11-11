//! The decision checkpoint is explicitly registered, with its own head and protocol.
use crate::core::tensor::{GGMLType, MetaValue, TensorSource};
use crate::models::qwen35::trunk::{HybridTrunk, Qwen35Config};
use crate::ops::kernel::{QuantizedTensor, Weight};

pub(super) fn string<'a>(source: &'a dyn TensorSource, key: &str) -> Result<&'a str, String> {
    source
        .metadata(key)
        .and_then(MetaValue::to_string_val)
        .ok_or_else(|| format!("Missing THX-02 metadata {key}"))
}

fn tensor(
    source: &dyn TensorSource,
    name: &str,
    dims: &[usize],
    vector: bool,
) -> Result<(), String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing THX-02 component tensor {name}"))?;
    if info.dims.iter().copied().ne(dims.iter().map(|&v| v as u64)) {
        return Err(format!(
            "THX-02 {name}: expected shape {dims:?}, got {:?}",
            info.dims
        ));
    }
    let data = source
        .tensor_slice(name)
        .ok_or_else(|| format!("THX-02 missing payload {name}"))?;
    if info.checked_nbytes() != Some(data.len() as u64)
        || !dims[0].is_multiple_of(info.ggml_type.type_traits().0)
    {
        return Err(format!(
            "THX-02 {name}: invalid payload length or block alignment"
        ));
    }
    let float = matches!(
        info.ggml_type,
        GGMLType::F32 | GGMLType::F16 | GGMLType::BF16
    );
    if !float
        && (vector
            || !matches!(
                info.ggml_type,
                GGMLType::Q8_0
                    | GGMLType::Q4_0
                    | GGMLType::Q4_1
                    | GGMLType::Q4K
                    | GGMLType::Q5K
                    | GGMLType::Q6K
            ))
    {
        return Err(format!(
            "THX-02 {name}: unsupported dtype {:?}",
            info.ggml_type
        ));
    }
    Ok(())
}

pub(super) fn config(source: &dyn TensorSource, thinker: bool) -> Result<Qwen35Config, String> {
    let arch = string(source, "general.architecture")?;
    if arch != if thinker { "qwen35" } else { super::ARCH }
        || string(source, "thx02.component")? != if thinker { "thinker" } else { "decision" }
        || string(source, "thx02.attention_mode")?
            != if thinker {
                "causal"
            } else {
                "noncausal_full_attention"
            }
        || string(source, "thx02.recurrent_head_layout")? != "qwen35-modulo-v1"
        || source
            .metadata("thx02.protocol_version")
            .and_then(MetaValue::to_u64)
            != Some(3)
    {
        return Err(
            "THX-02 architecture, component, attention, head layout or protocol mismatch".into(),
        );
    }
    let c = Qwen35Config::from_source_for_arch(source, arch)?;
    if c.n_embd != 2560
        || c.n_layer != 32
        || c.n_nextn != 0
        || c.n_head != 16
        || c.n_head_kv != 4
        || c.key_length != 256
        || c.value_length != 256
        || c.vocab_size != 248320
        || c.rope_freq_base != 1e7
        || c.norm_eps != 1e-6
        || c.n_ff != 9216
        || c.rope_dimension_count != 64
        || c.rope_dimension_sections != [11, 11, 10, 0]
        || c.ssm_d_state != 128
        || c.ssm_n_group != 16
        || c.ssm_dt_rank != 32
        || c.ssm_d_inner != 4096
        || c.ssm_d_conv != 4
        || c.moe.is_some()
        || c.is_recurrent
            .iter()
            .enumerate()
            .any(|(i, &r)| r != ((i + 1) % 4 != 0))
    {
        return Err("THX-02 requires the published Qwen3.5-4B backbone geometry".into());
    }
    tensor(
        source,
        "token_embd.weight",
        &[c.n_embd, c.vocab_size],
        false,
    )?;
    tensor(source, "output_norm.weight", &[c.n_embd], true)?;
    if !thinker {
        tensor(source, "cls.output.weight", &[c.n_embd, 255], false)?;
    }
    for i in 0..c.n_layer {
        let p = format!("blk.{i}");
        for name in ["attn_norm", "post_attention_norm"] {
            tensor(source, &format!("{p}.{name}.weight"), &[c.n_embd], true)?;
        }
        for (name, dims) in [
            ("ffn_gate", [c.n_embd, c.n_ff]),
            ("ffn_up", [c.n_embd, c.n_ff]),
            ("ffn_down", [c.n_ff, c.n_embd]),
        ] {
            tensor(source, &format!("{p}.{name}.weight"), &dims, false)?;
        }
        if c.is_recurrent[i] {
            for (name, dims) in [
                ("attn_qkv", [c.n_embd, 8192]),
                ("attn_gate", [c.n_embd, 4096]),
                ("ssm_alpha", [c.n_embd, 32]),
                ("ssm_beta", [c.n_embd, 32]),
                ("ssm_out", [4096, c.n_embd]),
            ] {
                tensor(source, &format!("{p}.{name}.weight"), &dims, false)?;
            }
            tensor(source, &format!("{p}.ssm_conv1d.weight"), &[4, 8192], true)?;
            for (name, n) in [("ssm_dt.bias", 32), ("ssm_a", 32), ("ssm_norm.weight", 128)] {
                tensor(source, &format!("{p}.{name}"), &[n], true)?;
            }
        } else {
            for (name, n) in [("attn_q", 8192), ("attn_k", 1024), ("attn_v", 1024)] {
                tensor(source, &format!("{p}.{name}.weight"), &[c.n_embd, n], false)?;
            }
            tensor(
                source,
                &format!("{p}.attn_output.weight"),
                &[4096, c.n_embd],
                false,
            )?;
            for name in ["attn_q_norm", "attn_k_norm"] {
                tensor(source, &format!("{p}.{name}.weight"), &[256], true)?;
            }
        }
    }
    Ok(c)
}

pub(super) fn load(
    source: &dyn TensorSource,
    config: Qwen35Config,
) -> Result<HybridTrunk<'_>, String> {
    let mut model = HybridTrunk::load_trunk(source, config, false)?;
    // THX's same-weight reference computes with F32 inputs; retain the shared SIMD BF16 kernel.
    fn bf16<'a>(weight: &mut Weight<'a>, source: &'a dyn TensorSource, name: &str) {
        if weight.ggml_type == GGMLType::BF16 {
            *weight = Weight::from_quantized(QuantizedTensor::from_bytes(
                source.tensor_slice(name).unwrap(),
                GGMLType::BF16,
                weight.n_in,
                weight.n_out,
            ));
        }
    }
    bf16(&mut model.tok_embd, source, "token_embd.weight");
    bf16(
        &mut model.output_weight,
        source,
        if source.tensor_info("output.weight").is_some() {
            "output.weight"
        } else {
            "token_embd.weight"
        },
    );
    if let Some(w) = &mut model.cls_score {
        bf16(w, source, "cls.output.weight");
    }
    for (i, layer) in model.layers.iter_mut().enumerate() {
        for (weight, name) in [
            (&mut layer.wq, "attn_q"),
            (&mut layer.wk, "attn_k"),
            (&mut layer.wv, "attn_v"),
            (&mut layer.wo, "attn_output"),
            (&mut layer.wqkv, "attn_qkv"),
            (&mut layer.wqkv_gate, "attn_gate"),
            (&mut layer.ssm_alpha, "ssm_alpha"),
            (&mut layer.ssm_beta, "ssm_beta"),
            (&mut layer.ssm_out, "ssm_out"),
            (&mut layer.ffn_gate, "ffn_gate"),
            (&mut layer.ffn_up, "ffn_up"),
            (&mut layer.ffn_down, "ffn_down"),
        ] {
            if let Some(weight) = weight {
                bf16(weight, source, &format!("blk.{i}.{name}.weight"));
            }
        }
    }
    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    #[derive(Default)]
    struct Empty {
        m: HashMap<String, MetaValue>,
    }
    impl TensorSource for Empty {
        fn metadata(&self, k: &str) -> Option<&MetaValue> {
            self.m.get(k)
        }
        fn tensor_info(&self, _: &str) -> Option<&crate::core::tensor::TensorInfo> {
            None
        }
        fn tensor_slice(&self, _: &str) -> Option<&[u8]> {
            None
        }
    }
    #[test]
    fn similar_architectures_are_never_silently_reinterpreted() {
        for arch in ["qwen35", "qwen35moe", "thx02", ""] {
            let mut source = Empty::default();
            source.m.insert(
                "general.architecture".into(),
                MetaValue::String(arch.into()),
            );
            assert!(config(&source, false).is_err());
            assert!(config(&source, true).is_err());
        }
    }

    #[test]
    fn tensor_contract_checks_every_trunk_storage_format_and_payload() {
        struct Source {
            info: crate::core::tensor::TensorInfo,
            bytes: Vec<u8>,
        }
        impl TensorSource for Source {
            fn metadata(&self, _: &str) -> Option<&MetaValue> {
                None
            }
            fn tensor_info(&self, _: &str) -> Option<&crate::core::tensor::TensorInfo> {
                Some(&self.info)
            }
            fn tensor_slice(&self, _: &str) -> Option<&[u8]> {
                Some(&self.bytes)
            }
        }
        for kind in [
            GGMLType::F32,
            GGMLType::F16,
            GGMLType::BF16,
            GGMLType::Q8_0,
            GGMLType::Q4_0,
            GGMLType::Q4_1,
            GGMLType::Q4K,
            GGMLType::Q5K,
            GGMLType::Q6K,
        ] {
            let info = crate::core::tensor::TensorInfo {
                name: "w".into(),
                dims: vec![256, 2],
                ggml_type: kind,
                offset: 0,
            };
            let mut source = Source {
                bytes: vec![0; info.checked_nbytes().unwrap() as usize],
                info,
            };
            assert!(tensor(&source, "w", &[256, 2], false).is_ok(), "{kind:?}");
            assert!(tensor(&source, "w", &[256, 3], false).is_err());
            source.bytes.pop();
            assert!(tensor(&source, "w", &[256, 2], false).is_err());
        }
    }
}
