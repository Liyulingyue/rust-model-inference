//! Qwen-Drive planning and perception heads.

pub mod config;
pub mod weights;

#[cfg(test)]
mod tests {
    use super::config::{PerceptionConfig, PlannerConfig};
    use super::weights::HeadLinear;
    use crate::core::tensor::{GGMLType, MetaValue, MetaValueType, TensorInfo, TensorSource};
    use std::collections::BTreeMap;

    #[derive(Default)]
    struct Source {
        metadata: BTreeMap<String, MetaValue>,
        tensors: BTreeMap<String, (TensorInfo, Vec<u8>)>,
    }

    impl TensorSource for Source {
        fn metadata(&self, key: &str) -> Option<&MetaValue> {
            self.metadata.get(key)
        }

        fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
            self.tensors.get(name).map(|(info, _)| info)
        }

        fn tensor_slice(&self, name: &str) -> Option<&[u8]> {
            self.tensors.get(name).map(|(_, bytes)| bytes.as_slice())
        }
    }

    fn planner_source() -> Source {
        let mut source = Source::default();
        source.metadata.insert(
            "general.architecture".into(),
            MetaValue::String("qwen_drive_planner".into()),
        );
        for (key, value) in [
            ("hidden_size", 1024),
            ("intermediate_size", 3584),
            ("num_hidden_layers", 32),
            ("num_attention_heads", 16),
            ("num_key_value_heads", 4),
            ("head_dim", 256),
            ("layers_per_kv", 4),
            ("num_future_points", 50),
            ("num_history_points", 16),
            ("trajectory_point_dim", 3),
            ("nav_command_classes", 3),
            ("ego_status_dim", 8),
            ("history_dynamics_dim", 2),
            ("time_embed_dim", 128),
            ("fourier_num_features", 16),
            ("num_inference_steps", 10),
            ("noise_seed", 42),
            ("max_reasoning_tokens", 256),
        ] {
            source.metadata.insert(
                format!("qwen_drive_planner.{key}"),
                MetaValue::Uint64(value),
            );
        }
        for (key, value) in [
            ("rms_norm_eps", 1e-5),
            ("time_embed_scale", 1000.0),
            ("fourier_max_frequency", 16.0),
            ("rope_theta", 10_000_000.0),
            ("min_one_minus_t", 0.1),
            ("noise_init_std", 1.0),
            ("trajectory_hz", 10.0),
        ] {
            source.metadata.insert(
                format!("qwen_drive_planner.{key}"),
                MetaValue::Float64(value),
            );
        }
        for (key, values) in [
            ("mrope_section", vec![11, 11, 10]),
            ("trajectory_scale", vec![165, 25, 1]),
        ] {
            source.metadata.insert(
                format!("qwen_drive_planner.{key}"),
                MetaValue::Array(
                    MetaValueType::Uint32,
                    values.into_iter().map(MetaValue::Uint32).collect(),
                ),
            );
        }
        source.metadata.insert(
            "qwen_drive_planner.trajectory_scale".into(),
            MetaValue::Array(
                MetaValueType::Float64,
                vec![165.0, 25.0, 1.5703125]
                    .into_iter()
                    .map(MetaValue::Float64)
                    .collect(),
            ),
        );
        source
    }

    #[test]
    fn qwen_drive_config_accepts_only_released_planner_geometry() {
        let mut source = planner_source();
        let config = PlannerConfig::from_source(&source).unwrap();
        assert_eq!(
            (config.hidden, config.layers, config.heads, config.kv_heads),
            (1024, 32, 16, 4)
        );
        source
            .metadata
            .insert("qwen_drive_planner.head_dim".into(), MetaValue::Uint64(128));
        assert!(PlannerConfig::from_source(&source)
            .unwrap_err()
            .contains("head_dim"));
    }

    #[test]
    fn qwen_drive_config_rejects_wrong_perception_architecture_first() {
        let error = PerceptionConfig::from_source(&planner_source()).unwrap_err();
        assert!(error.contains("qwen_drive_perception"));
    }

    #[test]
    fn qwen_drive_config_rounds_perception_f32_weights_to_bf16_compute() {
        let name = "qwen_drive_perception.test.weight";
        let values = [1.00390625f32, 1.01171875];
        let mut source = Source::default();
        source.tensors.insert(
            name.into(),
            (
                TensorInfo {
                    name: name.into(),
                    dims: vec![2, 1],
                    ggml_type: GGMLType::F32,
                    offset: 0,
                },
                values.into_iter().flat_map(f32::to_le_bytes).collect(),
            ),
        );
        let linear =
            HeadLinear::load_perception(&source, "qwen_drive_perception.test", 2, 1, false)
                .unwrap();
        assert_eq!(
            linear.forward_one(&[1.0, 1.0]).unwrap()[0].to_bits(),
            2.015625f32.to_bits()
        );
    }
}
