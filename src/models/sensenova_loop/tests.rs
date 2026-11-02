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
fn strict_vulkan_is_rejected() {
    use crate::compute::ComputePolicy;
    assert!(ComputePolicy::Cpu.validate_text_arch(ARCH).is_ok());
    assert!(ComputePolicy::Auto.validate_text_arch(ARCH).is_ok());
    assert!(ComputePolicy::Vulkan
        .validate_text_arch(ARCH)
        .unwrap_err()
        .contains("SenseNova"));
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
    assert!(check_tensor(&source, "router", &[4, 8]).is_ok());
    assert!(check_tensor(&source, "missing", &[4, 8]).is_err());
    assert!(check_tensor(&source, "router", &[8, 4]).is_err());
    source.bytes.pop();
    assert!(check_tensor(&source, "router", &[4, 8]).is_err());
    source.info.ggml_type = GGMLType::BF16;
    source.bytes = vec![0; 64];
    assert!(check_tensor(&source, "router", &[4, 8]).is_err());
}
