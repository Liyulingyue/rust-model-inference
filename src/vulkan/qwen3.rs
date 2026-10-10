use super::dense::{DenseLayer, DenseShape, DenseWeight, DenseWeights};
use super::{VulkanContext, VulkanError};
use crate::core::scratchpad::{KvCache, KvState};
use crate::core::tensor::GGMLType;
use crate::models::qwen3::trunk::{Qwen3Model, Qwen3Rope};

#[derive(Debug, Clone)]
pub(crate) struct EligibilityFacts {
    pub(crate) architecture: String,
    pub(crate) has_moe: bool,
    pub(crate) n_deepstack_layers: usize,
    pub(crate) has_qkv_bias: bool,
    pub(crate) rope: Qwen3Rope,
    pub(crate) weight_formats: Vec<GGMLType>,
    pub(crate) gate_up_formats: Vec<[GGMLType; 2]>,
}

pub(crate) fn check_eligibility(facts: &EligibilityFacts) -> Result<(), String> {
    if facts.architecture != "qwen3" {
        return Err(format!("unsupported architecture {}", facts.architecture));
    }
    if facts.has_moe {
        return Err("moe models are not supported".into());
    }
    if facts.n_deepstack_layers != 0 {
        return Err("deepstack models are not supported".into());
    }
    if facts.has_qkv_bias {
        return Err("qkv bias is not supported".into());
    }
    if facts.rope != Qwen3Rope::Neox {
        return Err("rope layout is not supported".into());
    }
    if facts.weight_formats.is_empty()
        || facts.weight_formats.iter().any(|&format| {
            !matches!(
                format,
                GGMLType::Q8_0
                    | GGMLType::Q4_0
                    | GGMLType::Q4_1
                    | GGMLType::Q4K
                    | GGMLType::Q6K
                    | GGMLType::F16
            )
        })
    {
        return Err("unsupported Vulkan weight format".into());
    }
    if facts
        .gate_up_formats
        .iter()
        .any(|formats| formats[0] != formats[1])
    {
        return Err("heterogeneous gate/up weight formats are not supported".into());
    }
    Ok(())
}

fn check_device_eligibility(shader_float16: bool) -> Result<(), String> {
    shader_float16
        .then_some(())
        .ok_or_else(|| "selected Vulkan device does not support shaderFloat16".into())
}

pub(crate) fn commit_shadow_kv(
    state: &mut KvState,
    position: usize,
    k_delta: &[f32],
    v_delta: &[f32],
) -> Result<(), String> {
    commit_shadow_kv_chunk(state, position, 1, k_delta, v_delta)
}

pub(crate) fn commit_shadow_kv_chunk(
    state: &mut KvState,
    position: usize,
    rows: usize,
    k_delta: &[f32],
    v_delta: &[f32],
) -> Result<(), String> {
    let stride = state
        .arch
        .n_head_kv
        .checked_mul(state.arch.n_embd_head_k.max(state.arch.n_embd_head_v))
        .ok_or("KV stride overflow")?;
    crate::compute::state::commit_kv_cache(
        &mut state.cache,
        state.arch.n_layer,
        state.capacity,
        stride,
        state.seq_len,
        position,
        rows,
        k_delta,
        v_delta,
    )?;
    state.seq_len = position + rows;
    state.update_access();
    Ok(())
}

pub(crate) use super::dense::{DenseVulkanSession as Qwen3VulkanSession, UploadedBuffers};

impl Qwen3VulkanSession {
    pub(crate) fn try_new(
        model: &Qwen3Model,
        capacity: usize,
        context: &'static VulkanContext,
    ) -> Result<Option<Self>, VulkanError> {
        if let Err(reason) = check_device_eligibility(context.supports_shader_float16())
            .and_then(|()| eligibility_facts(model).and_then(|facts| check_eligibility(&facts)))
        {
            eprintln!("[GPU] Qwen3 Vulkan unavailable: {reason}. Falling back to CPU.");
            return Ok(None);
        }
        let shape = dense_shape(model);
        Self::new(
            shape,
            &dense_weights(model)?,
            capacity,
            capacity.min(crate::core::prefill::DEFAULT_PREFILL_BATCH_SIZE),
            context,
        )
        .map(Some)
    }

    pub(crate) fn reserve_rows(
        &mut self,
        model: &Qwen3Model,
        rows: usize,
    ) -> Result<(), VulkanError> {
        self.reserve_dense_rows(&dense_weights(model)?, rows)
    }
}

fn dense_shape(model: &Qwen3Model) -> DenseShape {
    let c = &model.config;
    DenseShape {
        n_embd: c.n_embd,
        n_ff: c.n_ff,
        n_layer: c.n_layer,
        n_head: c.n_head,
        n_head_kv: c.n_head_kv,
        n_embd_head_k: c.n_embd_head_k,
        n_embd_head_v: c.n_embd_head_v,
        eps: c.eps,
        has_qk_norm: c.has_qk_norm,
        vocab: c.vocab,
        freq_base: c.freq_base,
        rope_layout: super::ops::RopeLayout::Neox,
        approximate_silu_multiline: true,
        attention_mode: crate::vulkan::ops::AttentionMode::PreparedF16,
    }
}
fn dense_weights(model: &Qwen3Model) -> Result<DenseWeights<'_>, VulkanError> {
    let view = |name: &str,
                weight: &crate::ops::kernel::Weight<'_>|
     -> Result<DenseWeight<'_>, VulkanError> {
        let bytes = model
            .source
            .tensor_slice(name)
            .ok_or_else(|| VulkanError::UnsupportedShape(format!("missing Qwen3 tensor {name}")))?;
        Ok(DenseWeight {
            bytes,
            ggml_type: weight.ggml_type,
            n_in: weight.n_in,
            n_out: weight.n_out,
        })
    };
    let layers = model
        .layers
        .iter()
        .enumerate()
        .map(|(i, l)| {
            Ok(DenseLayer {
                attn_norm: &l.attn_norm,
                ffn_norm: &l.ffn_norm,
                q_norm: l.q_norm.as_deref(),
                k_norm: l.k_norm.as_deref(),
                wq: view(&format!("blk.{i}.attn_q.weight"), &l.wq)?,
                wk: view(&format!("blk.{i}.attn_k.weight"), &l.wk)?,
                wv: view(&format!("blk.{i}.attn_v.weight"), &l.wv)?,
                wo: view(&format!("blk.{i}.attn_output.weight"), &l.wo)?,
                w_gate: view(&format!("blk.{i}.ffn_gate.weight"), &l.w_gate)?,
                w_up: view(&format!("blk.{i}.ffn_up.weight"), &l.w_up)?,
                w_down: view(&format!("blk.{i}.ffn_down.weight"), &l.w_down)?,
            })
        })
        .collect::<Result<_, VulkanError>>()?;
    Ok(DenseWeights {
        layers,
        output_norm: &model.output_norm,
        output: view(output_tensor_name(model), &model.output)?,
    })
}

fn eligibility_facts(model: &Qwen3Model) -> Result<EligibilityFacts, String> {
    let mut weight_formats = Vec::with_capacity(model.config.n_layer * 7 + 1);
    let mut gate_up_formats = Vec::with_capacity(model.config.n_layer);
    for layer in 0..model.config.n_layer {
        for suffix in [
            "attn_q.weight",
            "attn_k.weight",
            "attn_v.weight",
            "attn_output.weight",
            "ffn_gate.weight",
            "ffn_up.weight",
            "ffn_down.weight",
        ] {
            let name = format!("blk.{layer}.{suffix}");
            let info = model
                .source
                .tensor_info(&name)
                .ok_or_else(|| format!("missing Qwen3 tensor {name}"))?;
            weight_formats.push(info.ggml_type);
        }
        gate_up_formats.push([
            model.layers[layer].w_gate.ggml_type,
            model.layers[layer].w_up.ggml_type,
        ]);
    }
    let output_name = output_tensor_name(model);
    weight_formats.push(
        model
            .source
            .tensor_info(output_name)
            .ok_or_else(|| format!("missing Qwen3 tensor {output_name}"))?
            .ggml_type,
    );
    Ok(EligibilityFacts {
        architecture: model.config.architecture.clone(),
        has_moe: model.config.moe.is_some(),
        n_deepstack_layers: model.config.n_deepstack_layers,
        has_qkv_bias: model.config.has_qkv_bias,
        rope: model.config.rope,
        weight_formats,
        gate_up_formats,
    })
}

fn output_tensor_name(model: &Qwen3Model) -> &'static str {
    if model.source.tensor_info("output.weight").is_some() {
        "output.weight"
    } else {
        "token_embd.weight"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::scratchpad::{KvArch, KvCache, KvFormat, KvState};
    use crate::core::tensor::GGMLType;
    use crate::models::qwen3::trunk::Qwen3Rope;
    use std::sync::Arc;

    fn eligible_facts() -> EligibilityFacts {
        EligibilityFacts {
            architecture: "qwen3".into(),
            has_moe: false,
            n_deepstack_layers: 0,
            has_qkv_bias: false,
            rope: Qwen3Rope::Neox,
            weight_formats: vec![GGMLType::Q8_0; 198],
            gate_up_formats: vec![[GGMLType::Q8_0, GGMLType::Q8_0]; 28],
        }
    }

    #[test]
    fn qwen3_q8_dense_is_eligible() {
        let facts = eligible_facts();
        assert_eq!(check_eligibility(&facts), Ok(()));
    }

    #[test]
    fn qwen3_full_model_requires_shader_float16() {
        assert!(check_device_eligibility(false)
            .expect_err("full-model Qwen3 must stay on CPU without shaderFloat16")
            .contains("shaderFloat16"));
    }

    #[test]
    fn heterogeneous_gate_up_stays_on_cpu_before_session_initialization() {
        let mut facts = eligible_facts();
        facts.gate_up_formats[7] = [GGMLType::Q4K, GGMLType::Q6K];

        assert!(check_eligibility(&facts)
            .expect_err("heterogeneous gate/up must be rejected by preflight")
            .contains("heterogeneous gate/up"));
    }

    #[test]
    fn unsupported_architecture_stays_on_cpu() {
        let mut facts = eligible_facts();
        facts.architecture = "qwen3vl".into();
        assert!(check_eligibility(&facts).is_err());
    }

    #[test]
    fn unsupported_dense_facts_stay_on_cpu() {
        let cases = [
            (
                "moe",
                EligibilityFacts {
                    has_moe: true,
                    ..eligible_facts()
                },
            ),
            (
                "deepstack",
                EligibilityFacts {
                    n_deepstack_layers: 4,
                    ..eligible_facts()
                },
            ),
            (
                "qkv bias",
                EligibilityFacts {
                    has_qkv_bias: true,
                    ..eligible_facts()
                },
            ),
            (
                "rope",
                EligibilityFacts {
                    rope: Qwen3Rope::Interleaved {
                        sections: [16, 24, 24, 0],
                        n_dims: 64,
                    },
                    ..eligible_facts()
                },
            ),
            (
                "weight format",
                EligibilityFacts {
                    weight_formats: vec![GGMLType::BF16],
                    ..eligible_facts()
                },
            ),
            (
                "weight format",
                EligibilityFacts {
                    weight_formats: vec![GGMLType::Q5K],
                    ..eligible_facts()
                },
            ),
            (
                "weight format",
                EligibilityFacts {
                    weight_formats: Vec::new(),
                    ..eligible_facts()
                },
            ),
        ];
        for (message, facts) in cases {
            assert!(check_eligibility(&facts)
                .expect_err("unsupported facts must be rejected")
                .contains(message));
        }
    }

    #[test]
    fn shadow_kv_chunk_commits_all_layers_and_rows_atomically() {
        for format in [KvFormat::F16, KvFormat::F32] {
            let arch = Arc::new(KvArch::new(2, 1, 2, 2, 4));
            let mut state = KvState::new(arch, format, 4);
            commit_shadow_kv_chunk(&mut state, 0, 1, &[31.0; 4], &[32.0; 4]).unwrap();
            let k = [
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ];
            let v = k.map(|value| value + 12.0);
            let snapshot = |state: &KvState| match &state.cache {
                KvCache::F16(cache) => (
                    cache.k.iter().map(|&v| v as u32).collect::<Vec<_>>(),
                    cache.v.iter().map(|&v| v as u32).collect::<Vec<_>>(),
                ),
                KvCache::F32(cache) => (
                    cache.k.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    cache.v.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                ),
            };
            let before = snapshot(&state);
            assert!(commit_shadow_kv_chunk(&mut state, 1, 3, &k[..11], &v).is_err());
            let mut nonfinite = k;
            nonfinite[11] = f32::NAN;
            assert!(commit_shadow_kv_chunk(&mut state, 1, 3, &nonfinite, &v).is_err());
            assert!(commit_shadow_kv_chunk(&mut state, 0, 3, &k, &v).is_err());
            assert_eq!(state.seq_len, 1);
            assert_eq!(snapshot(&state), before);
            commit_shadow_kv_chunk(&mut state, 1, 3, &k, &v).unwrap();
            let expected = |values: &[f32]| {
                values
                    .iter()
                    .map(|&value| match format {
                        KvFormat::F16 => crate::ops::f32_to_f16(value) as u32,
                        KvFormat::F32 => value.to_bits(),
                    })
                    .collect::<Vec<_>>()
            };
            let (keys, values) = snapshot(&state);
            assert_eq!(&keys[2..8], &expected(&k[..6]));
            assert_eq!(&keys[10..16], &expected(&k[6..]));
            assert_eq!(&values[2..8], &expected(&v[..6]));
            assert_eq!(&values[10..16], &expected(&v[6..]));
            assert_eq!(state.seq_len, 4);
        }
    }
}
