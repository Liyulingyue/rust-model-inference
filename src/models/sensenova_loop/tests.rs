use super::*;

fn contract() -> serde_json::Value {
    serde_json::json!({"text_config":{"num_hidden_layers":8,"layer_types":["linear_attention","linear_attention","linear_attention","full_attention","linear_attention","linear_attention","linear_attention","full_attention"]},"loop_settings":{"start":2,"end":6,"eval_count":2,"counts":[2]},"mhc":{"start":2,"end":6,"streams":2,"iterations":20,"fixed_identity":false,"pre_scale":1.0}})
}
#[test]
fn visits_share_weights_but_have_distinct_cache_slots() {
    let (plan, _) = parse_contract(&contract().to_string(), 8).unwrap();
    assert_eq!(plan, vec![0, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 7]);
    assert_ne!(
        plan.iter().position(|v| *v == 3),
        plan.iter().rposition(|v| *v == 3)
    );
}
#[test]
fn rejects_cross_phase_mhc_and_excessive_visits() {
    let mut c = contract();
    c["mhc"]["end"] = 7.into();
    assert!(parse_contract(&c.to_string(), 8).is_err());
    let mut c = contract();
    c["loop_settings"]["eval_count"] = 100000.into();
    assert!(parse_contract(&c.to_string(), 8).is_err());
}
#[test]
fn sinkhorn_is_doubly_stochastic() {
    let mixed = sinkhorn(vec![3.0, 0.0, -1.0, 2.0], 2, 20);
    for i in 0..2 {
        assert!((mixed[i * 2] + mixed[i * 2 + 1] - 1.0).abs() < 1e-5);
        assert!((mixed[i] + mixed[i + 2] - 1.0).abs() < 1e-5);
    }
}
#[test]
fn stream_merge_preserves_row_order() {
    let merged = merge(
        &[1., 2., 10., 20.],
        &[3., 4.],
        &[0.5, 2.],
        &[1., 0., 0., 1.],
        2,
        2,
    );
    assert_eq!(merged, vec![2.5, 4., 16., 28.]);
}

#[test]
fn vulkan_has_an_explicit_decoder_path() {
    use crate::compute::ComputePolicy;
    assert!(ComputePolicy::Cpu.validate_text_arch(ARCH).is_ok());
    assert!(ComputePolicy::Auto.validate_text_arch(ARCH).is_ok());
    assert!(ComputePolicy::Vulkan.validate_text_arch(ARCH).is_ok());
    assert!(ComputePolicy::has_complete_device_path(ARCH));
}

#[test]
fn rejects_truncated_wrong_shape_or_bf16_router() {
    use crate::core::tensor::{MetaValue, TensorInfo};
    struct Source {
        info: TensorInfo,
        bytes: Vec<u8>,
    }
    impl TensorSource for Source {
        fn metadata(&self, _: &str) -> Option<&MetaValue> {
            None
        }
        fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
            (name == self.info.name).then_some(&self.info)
        }
        fn tensor_slice(&self, name: &str) -> Option<&[u8]> {
            (name == self.info.name).then_some(self.bytes.as_slice())
        }
    }
    let mut source = Source {
        info: TensorInfo {
            name: "router".into(),
            dims: vec![4, 8],
            ggml_type: GGMLType::F32,
            offset: 0,
        },
        bytes: vec![0; 128],
    };
    assert!(check_tensor(&source, "router", &[4, 8], true).is_ok());
    assert!(check_tensor(&source, "missing", &[4, 8], true).is_err());
    assert!(check_tensor(&source, "router", &[8, 4], true).is_err());
    source.bytes.pop();
    assert!(check_tensor(&source, "router", &[4, 8], true).is_err());
    // A projection may ship block-quantized or half precision: `load_weight`
    // decodes both, so the validator accepts them.
    source.info.ggml_type = GGMLType::BF16;
    source.bytes = vec![0; 64];
    assert!(check_tensor(&source, "router", &[4, 8], true).is_ok());
    // Q8_0 quantizes the contiguous row width, so dims[0] has to be a multiple
    // of the 32-element block: 32 values + one f16 scale per 4 rows.
    source.info.dims = vec![32, 4];
    source.info.ggml_type = GGMLType::Q8_0;
    source.bytes = vec![0; 136];
    assert!(check_tensor(&source, "router", &[32, 4], true).is_ok());
    // Norms, biases and the recurrent state are not block-quantized.
    source.info.dims = vec![4, 8];
    source.info.ggml_type = GGMLType::BF16;
    assert!(check_tensor(&source, "router", &[4, 8], false).is_err());
    source.info.dims = vec![32, 4];
    source.info.ggml_type = GGMLType::Q8_0;
    assert!(check_tensor(&source, "router", &[32, 4], false).is_err());
}

#[cfg(feature = "vulkan")]
mod gpu_tests {
    use super::*;
    use crate::compute::ComputePolicy;
    use crate::models::qwen35::trunk::HybridSession;
    use crate::ops::kernel::QuantizedTensor;
    use crate::vulkan::qwen35::{commit_shadow_state_chunk, Qwen35VulkanSession};
    use std::sync::Arc;

    type Session<'a> = HybridSession<'a, 'static, SenseNovaLoopModel<'static>>;

    fn weight(n_in: usize, n_out: usize, seed: usize) -> Weight<'static> {
        Weight::from_quantized(QuantizedTensor::F32 {
            data: (0..n_in * n_out)
                .map(|i| ((i * seed % 23) as f32 - 11.0) / 41.0)
                .collect(),
            n_in,
            n_out,
        })
    }

    fn tiny_loop_model() -> SenseNovaLoopModel<'static> {
        let mut trunk = crate::models::qwen35::trunk::tests::tiny_dense_session_model();
        let mut recurrent = crate::models::qwen35::trunk::tests::tiny_dense_session_model();
        for (index, layer) in [&mut trunk.layers[0], &mut recurrent.layers[0]]
            .into_iter()
            .enumerate()
        {
            layer.ffn_gate = Some(weight(4, 4, 3 + index));
            layer.ffn_up = Some(weight(4, 4, 7 + index));
            layer.ffn_down = Some(weight(4, 4, 11 + index));
        }
        let dense = &mut trunk.layers[0];
        dense.wq = Some(weight(4, 8, 3));
        dense.wk = Some(weight(4, 4, 5));
        dense.wv = Some(weight(4, 4, 7));
        dense.wo = Some(weight(4, 4, 13));
        let layer = &mut recurrent.layers[0];
        layer.wqkv = Some(weight(4, 16, 17));
        layer.wqkv_gate = Some(weight(4, 8, 19));
        layer.ssm_beta = Some(weight(4, 4, 7));
        layer.ssm_alpha = Some(weight(4, 4, 11));
        layer.ssm_conv1d = Some((0..32).map(|i| 0.1 + (i % 7) as f32 * 0.03).collect());
        layer.ssm_dt = Some(vec![-0.1, 0.2, 0.1, 0.0]);
        layer.ssm_a = Some(vec![-0.4, -0.2, -0.3, -0.1]);
        layer.ssm_norm = Some(vec![0.9, 1.1]);
        layer.ssm_out = Some(weight(8, 4, 13));
        trunk.layers.extend(recurrent.layers);
        trunk.output_weight = weight(4, 8, 17);
        trunk.config.n_layer = 4;
        trunk.config.is_recurrent = vec![false, true, false, true];
        trunk.config.rope_dimension_sections = [1, 1, 0, 0];
        trunk.config.ssm_d_conv = 2;
        trunk.config.ssm_n_group = 2;
        trunk.config.ssm_dt_rank = 4;
        trunk.config.ssm_d_inner = 8;
        let settings = MhcSettings {
            start: 0,
            end: 2,
            streams: 2,
            iterations: 7,
            fixed_identity: false,
            pre_scale: 0.75,
        };
        let routers = (0..2)
            .map(|physical| {
                let router = |seed| Router {
                    weight: weight(8, 8, seed),
                    bias: (0..8).map(|i| (i as f32 - 3.0) * 0.07).collect(),
                };
                (router(3 + physical * 2), router(7 + physical * 2))
            })
            .collect();
        SenseNovaLoopModel {
            trunk,
            settings,
            routers,
            plan: vec![0, 1, 0, 1],
        }
    }

    fn state(session: &Session<'_>) -> Vec<f32> {
        shadow_state(
            session.kv_cache(),
            &session.scratch().conv_states,
            &session.scratch().ssm_states,
        )
    }

    fn shadow_state(kv: &KvCache, conv: &[Vec<f32>], ssm: &[Vec<f32>]) -> Vec<f32> {
        let KvCache::F32(cache) = kv else {
            panic!("SenseNova session requires an F32 shadow cache")
        };
        cache
            .k
            .iter()
            .chain(&cache.v)
            .chain(conv.iter().flatten())
            .chain(ssm.iter().flatten())
            .copied()
            .collect()
    }

    fn close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        // CPU SIMD and Vulkan use different F32 reduction orders.
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                actual.is_finite()
                    && expected.is_finite()
                    && (actual - expected).abs() <= 2e-4 + 2e-4 * expected.abs(),
                "at {index}: actual={actual} expected={expected}"
            );
        }
    }

    #[test]
    fn full_graph_shader_prefill_decode_cache_and_reset_allows_software_icd() {
        crate::ops::enable_gpu();
        let Some(context) = crate::ops::get_vulkan_context() else {
            eprintln!("skip SenseNova shader graph: no Vulkan device");
            return;
        };
        if !context.supports_shader_float16() {
            eprintln!("skip SenseNova shader graph: requires shaderFloat16");
            return;
        }
        // Direct construction permits software ICDs for shader semantics only.
        // Production ComputePolicy continues to require a physical device.
        let mut cpu_model = tiny_loop_model();
        let gpu_model = tiny_loop_model();
        let cfg = gpu_model.trunk.config.clone();
        let capacity = 8;
        let mut cpu = HybridSession::new_with_compute(
            &mut cpu_model,
            capacity,
            3,
            Arc::new(ComputePool::new(1)),
            ComputePolicy::Cpu,
        )
        .unwrap();
        let mut gpu = Qwen35VulkanSession::try_new_rows_with_routing(
            &gpu_model.trunk,
            capacity,
            3,
            context,
            Some(StreamRouting {
                plan: &gpu_model.plan,
                settings: &gpu_model.settings,
                routers: &gpu_model.routers,
            }),
        )
        .unwrap()
        .expect("tiny SenseNova shader graph must be supported");
        let mut kv = KvCache::new_f32(cfg.n_layer_impl(), capacity, cfg.n_embd_gqa());
        let mut conv = cpu.scratch().conv_states.clone();
        let mut ssm = cpu.scratch().ssm_states.clone();
        let kv_stride = cfg.n_embd_head() * cfg.n_head_kv;
        let tokens = [0, 1, 2, 3, 4, 5];
        let positions: Vec<_> = (0..tokens.len())
            .map(|i| [i, i + 3, 2 * i + 7, 0])
            .collect();
        for replay in 0..2 {
            if replay != 0 {
                cpu.reset();
                gpu.reset().unwrap();
                kv = KvCache::new_f32(cfg.n_layer_impl(), capacity, cfg.n_embd_gqa());
                conv.iter_mut().for_each(|layer| layer.fill(0.0));
                ssm.iter_mut().for_each(|layer| layer.fill(0.0));
                assert!(state(&cpu).iter().all(|&v| v == 0.0));
                assert!(shadow_state(&kv, &conv, &ssm).iter().all(|&v| v == 0.0));
                assert_eq!(cpu.processed_tokens(), 0);
                assert_eq!(cpu.next_position(), 0);
            }
            for range in [0..3, 3..5, 5..6] {
                let input = cpu.embed_tokens(&tokens[range.clone()]).unwrap();
                let expected = cpu
                    .step_with_tokens(&tokens[range.clone()], &positions[range.clone()])
                    .unwrap();
                let result = gpu
                    .forward_chunk(&input, range.start, &positions[range.clone()], range.len())
                    .unwrap();
                close(result.logits, &expected);
                commit_shadow_state_chunk(
                    &mut kv,
                    &mut conv,
                    &mut ssm,
                    range.start,
                    range.len(),
                    capacity,
                    kv_stride,
                    result.k_delta,
                    result.v_delta,
                    result.conv_state,
                    result.ssm_state,
                )
                .unwrap();
                gpu.commit_token();
                close(&shadow_state(&kv, &conv, &ssm), &state(&cpu));
                assert_eq!(cpu.processed_tokens(), range.end);
                assert_eq!(cpu.next_position(), range.end);
                assert!(ssm[1].iter().any(|&v| v != 0.0) && ssm[3].iter().any(|&v| v != 0.0));
                assert_ne!(ssm[1], ssm[3]);
                assert_ne!(conv[1], conv[3]);
            }
        }
    }

    #[test]
    fn full_graph_gpu_prefill_decode_visit_state_fallback_and_reset() {
        crate::ops::enable_gpu();
        let Some(context) = crate::ops::get_vulkan_context() else {
            eprintln!("skip SenseNova full graph: no Vulkan device");
            return;
        };
        if context.is_software_icd() || !context.supports_shader_float16() {
            eprintln!(
                "skip SenseNova full graph: requires a physical Vulkan device with shaderFloat16"
            );
            return;
        }
        let mut cpu_model = tiny_loop_model();
        let mut gpu_model = tiny_loop_model();
        let mut auto_model = tiny_loop_model();
        let pool = Arc::new(ComputePool::new(1));
        let mut cpu =
            HybridSession::new_with_compute(&mut cpu_model, 8, 3, pool.clone(), ComputePolicy::Cpu)
                .unwrap();
        let mut gpu = HybridSession::new_with_compute(
            &mut gpu_model,
            8,
            3,
            pool.clone(),
            ComputePolicy::Vulkan,
        )
        .unwrap();
        let mut auto =
            HybridSession::new_with_compute(&mut auto_model, 8, 3, pool, ComputePolicy::Auto)
                .unwrap();
        assert!(gpu.gpu_enabled_for_test() && auto.gpu_enabled_for_test());
        let tokens = [0, 1, 2, 3, 4, 5];
        let positions: Vec<_> = (0..tokens.len())
            .map(|i| [i, i + 3, 2 * i + 7, 0])
            .collect();
        for (step, range) in [0..3, 3..5, 5..6].into_iter().enumerate() {
            if step == 1 {
                // Fail after device work has mutated a chunk, keeping its committed prefix.
                auto.fail_gpu_after_row_for_test(0);
            }
            let expected = cpu
                .step_with_tokens(&tokens[range.clone()], &positions[range.clone()])
                .unwrap();
            let actual = gpu
                .step_with_tokens(&tokens[range.clone()], &positions[range.clone()])
                .unwrap();
            let fallback = auto
                .step_with_tokens(&tokens[range.clone()], &positions[range.clone()])
                .unwrap();
            close(&actual, &expected);
            close(&fallback, &expected);
            close(&state(&gpu), &state(&cpu));
            close(&state(&auto), &state(&cpu));
            assert!(gpu.gpu_enabled_for_test());
            assert_eq!(auto.gpu_enabled_for_test(), step == 0);
            for session in [&cpu, &gpu, &auto] {
                assert_eq!(session.processed_tokens(), range.end);
                assert_eq!(session.next_position(), range.end);
                let dense = session.dense_kv_snapshots(&[0, 2], range.end).unwrap();
                assert!(dense.iter().all(|slot| slot
                    .key
                    .iter()
                    .chain(&slot.value)
                    .any(|&v| v != 0.0)));
                assert_ne!(
                    dense[0].value, dense[1].value,
                    "repeated dense visits need separate caches"
                );
                let ssm = &session.scratch().ssm_states;
                let conv = &session.scratch().conv_states;
                assert!(ssm[1].iter().any(|&v| v != 0.0) && ssm[3].iter().any(|&v| v != 0.0));
                assert_ne!(
                    ssm[1], ssm[3],
                    "repeated recurrent visits need separate state"
                );
                assert_ne!(conv[1], conv[3]);
            }
        }
        for session in [&mut cpu, &mut gpu, &mut auto] {
            session.reset();
            assert_eq!(session.processed_tokens(), 0);
            assert_eq!(session.next_position(), 0);
            assert!(state(session).iter().all(|&v| v == 0.0));
        }
        assert!(gpu.gpu_enabled_for_test());
        assert!(
            !auto.gpu_enabled_for_test(),
            "failed Auto executor stays disabled after reset"
        );
        let expected = cpu.step_with_tokens(&tokens[..3], &positions[..3]).unwrap();
        close(
            &gpu.step_with_tokens(&tokens[..3], &positions[..3]).unwrap(),
            &expected,
        );
        close(
            &auto
                .step_with_tokens(&tokens[..3], &positions[..3])
                .unwrap(),
            &expected,
        );
        close(&state(&gpu), &state(&cpu));
        close(&state(&auto), &state(&cpu));
    }
}
