//! Qwen-Drive planning and perception heads.

pub mod config;
pub mod perception;
pub mod planning;
pub mod rng;
pub mod scene;
pub mod weights;

#[cfg(test)]
mod tests {
    use super::config::{PerceptionConfig, PlannerConfig};
    use super::perception::bev::BevFormer;
    use super::perception::fpn::{
        conv_transpose2d, depth_softmax, frustum_voxel_coordinates, layer_norm_2d,
        voxel_to_bev_tokens, ViewBackbone, ViewGeometry,
    };
    use super::perception::heads::PerceptionHeads;
    use super::perception::ops::{
        grid_sample_bilinear, ms_deform_attn, resize_bilinear, resize_bilinear_aligned,
        voxel_pool_depth, DeformAttentionInput, Tensor4, VoxelPoolInput,
    };
    use super::perception::QwenDrivePerception;
    use super::planning::{
        euler_update, fourier_features, normalize_history, time_embedding, validate_sample_request,
        waypoint_mrope, PlanningExpert,
    };
    use super::rng::TorchNormalRng;
    use super::scene::PlanningScene;
    use super::weights::HeadLinear;
    use crate::core::tensor::{GGMLType, MetaValue, MetaValueType, TensorInfo, TensorSource};
    use serde::Deserialize;
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

    #[test]
    fn qwen_drive_planner_linear_rounds_bf16_compute_output() {
        let name = "qwen_drive_planner.test.weight";
        let mut source = Source::default();
        source.tensors.insert(
            name.into(),
            (
                TensorInfo {
                    name: name.into(),
                    dims: vec![2, 1],
                    ggml_type: GGMLType::BF16,
                    offset: 0,
                },
                [0x3e9a_u16, 0x3f33_u16]
                    .into_iter()
                    .flat_map(u16::to_le_bytes)
                    .collect(),
            ),
        );
        let linear = HeadLinear::load(&source, "qwen_drive_planner.test", 2, 1, false).unwrap();
        assert_eq!(
            linear
                .forward_one(&[f32::from_bits(0x3e4d_0000), f32::from_bits(0x3ecd_0000)])
                .unwrap()[0]
                .to_bits(),
            0x3eae_0000
        );
    }

    #[derive(Deserialize)]
    struct PlannerOperators {
        time_inputs: Vec<u32>,
        time: Vec<u32>,
        point_inputs: Vec<u32>,
        fourier: Vec<u32>,
        rope_input: Vec<u32>,
        rope: Vec<u32>,
        history_input: Vec<u32>,
        history: Vec<u32>,
        euler_input: Vec<u32>,
        euler_endpoint: Vec<u32>,
        euler: Vec<u32>,
        normal: BTreeMap<String, Vec<u32>>,
    }

    #[derive(Deserialize)]
    struct TensorFixture {
        shape: [usize; 4],
        values: Vec<u32>,
    }

    #[derive(Deserialize)]
    struct ResizeFixture {
        input: TensorFixture,
        output_hw: [usize; 2],
        output: Vec<u32>,
    }

    #[derive(Deserialize)]
    struct GridFixture {
        input: TensorFixture,
        grid_shape: [usize; 3],
        grid: Vec<u32>,
        output: Vec<u32>,
    }

    #[derive(Deserialize)]
    struct VoxelFixture {
        img_feats: Vec<u32>,
        img_depth: Vec<u32>,
        coords: Vec<[usize; 4]>,
        point_indices: Vec<usize>,
        shape: [usize; 10],
        output: Vec<u32>,
    }

    #[derive(Deserialize)]
    struct DeformFixture {
        value: Vec<u32>,
        spatial_shapes: Vec<[usize; 2]>,
        level_start_index: Vec<usize>,
        sampling_locations: Vec<u32>,
        attention_weights: Vec<u32>,
        shape: [usize; 5],
        output: Vec<u32>,
    }

    #[derive(Deserialize)]
    struct PerceptionOperators {
        resize: ResizeFixture,
        grid: GridFixture,
        voxel: VoxelFixture,
        deform: DeformFixture,
    }

    #[derive(Deserialize)]
    struct NormFixture {
        input: TensorFixture,
        weight: Vec<u32>,
        bias: Vec<u32>,
        epsilon: u32,
        output: Vec<u32>,
    }

    #[derive(Deserialize)]
    struct TransposeFixture {
        input: TensorFixture,
        weight_shape: [usize; 4],
        weight: Vec<u32>,
        bias: Vec<u32>,
        stride: [usize; 2],
        padding: [usize; 2],
        output_shape: [usize; 4],
        output: Vec<u32>,
    }

    #[derive(Deserialize)]
    struct GeometryFixture {
        frustum_range: Vec<u32>,
        frustum_size: Vec<u32>,
        pc_range: Vec<u32>,
        voxel_size: Vec<u32>,
        voxel_shape: [usize; 3],
        lidar2img: Vec<Vec<u32>>,
        lidar2ego: Vec<Vec<u32>>,
        coords: Vec<[usize; 4]>,
        point_indices: Vec<usize>,
    }

    #[derive(Deserialize)]
    struct BevFixture {
        values: Vec<u32>,
        shape: [usize; 5],
        weight: Vec<u32>,
        weight_shape: [usize; 2],
        bias: Vec<u32>,
        output: Vec<u32>,
    }

    #[derive(Deserialize)]
    struct PerceptionViewFixture {
        layer_norm: NormFixture,
        transpose: TransposeFixture,
        resize_aligned: ResizeFixture,
        depth: TensorFixture,
        depth_output: Vec<u32>,
        geometry: GeometryFixture,
        bev: BevFixture,
    }

    fn values(bits: &[u32]) -> Vec<f32> {
        bits.iter().copied().map(f32::from_bits).collect()
    }

    fn assert_bits(label: &str, actual: &[f32], expected: &[u32]) {
        assert_eq!(actual.len(), expected.len(), "{label} length");
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected,
                "{label}[{index}] rust={:08x} oracle={expected:08x}",
                actual.to_bits()
            );
        }
    }

    #[test]
    fn planner_scalar_fixture_matches_every_f32_word() {
        let fixture: PlannerOperators = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/qwen_drive/planner-operators.json"
        )))
        .unwrap();
        assert_bits(
            "time",
            &time_embedding(&values(&fixture.time_inputs), 128, 1000.0).unwrap(),
            &fixture.time,
        );
        assert_bits(
            "fourier",
            &fourier_features(&values(&fixture.point_inputs), 3, 16, 16.0).unwrap(),
            &fixture.fourier,
        );
        assert_bits(
            "rope",
            &waypoint_mrope(
                &values(&fixture.rope_input),
                2,
                2,
                8,
                6,
                [1, 1, 1],
                [257, 258, 259],
                10_000_000.0,
            )
            .unwrap(),
            &fixture.rope,
        );
        assert_bits(
            "history",
            &normalize_history(
                &values(&fixture.history_input),
                16,
                [165.0, 25.0, 1.5703125],
            )
            .unwrap(),
            &fixture.history,
        );
        assert_bits(
            "euler",
            &euler_update(
                &values(&fixture.euler_input),
                &values(&fixture.euler_endpoint),
                0.8,
                0.1,
                0.1,
            )
            .unwrap(),
            &fixture.euler,
        );
        for seed in [42_u64, 43] {
            assert_bits(
                &format!("normal seed {seed}"),
                &TorchNormalRng::normal_f32(seed, 150),
                &fixture.normal[&seed.to_string()],
            );
        }
    }

    #[test]
    fn perception_primitives_match_scalar_oracle_words() {
        let fixture: PerceptionOperators = serde_json::from_str(
            &std::fs::read_to_string(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("tests/fixtures/qwen_drive/perception-operators.json"),
            )
            .unwrap(),
        )
        .unwrap();
        let tensor =
            |fixture: &TensorFixture| Tensor4::new(values(&fixture.values), fixture.shape).unwrap();
        assert_bits(
            "resize",
            resize_bilinear(
                &tensor(&fixture.resize.input),
                fixture.resize.output_hw[0],
                fixture.resize.output_hw[1],
            )
            .unwrap()
            .values(),
            &fixture.resize.output,
        );
        assert_bits(
            "grid",
            grid_sample_bilinear(
                &tensor(&fixture.grid.input),
                &values(&fixture.grid.grid),
                fixture.grid.grid_shape,
            )
            .unwrap()
            .values(),
            &fixture.grid.output,
        );
        let [batch, sweeps, cameras, x, y, z, depth, height, width, channels] = fixture.voxel.shape;
        assert_bits(
            "voxel",
            &voxel_pool_depth(&VoxelPoolInput {
                img_feats: &values(&fixture.voxel.img_feats),
                img_depth: &values(&fixture.voxel.img_depth),
                coords: &fixture.voxel.coords,
                point_indices: &fixture.voxel.point_indices,
                batch,
                sweeps,
                cameras,
                x,
                y,
                z,
                depth,
                height,
                width,
                channels,
            })
            .unwrap(),
            &fixture.voxel.output,
        );
        let [batch, queries, heads, channels, points] = fixture.deform.shape;
        assert_bits(
            "deform",
            &ms_deform_attn(&DeformAttentionInput {
                value: &values(&fixture.deform.value),
                spatial_shapes: &fixture.deform.spatial_shapes,
                level_start_index: &fixture.deform.level_start_index,
                sampling_locations: &values(&fixture.deform.sampling_locations),
                attention_weights: &values(&fixture.deform.attention_weights),
                batch,
                queries,
                heads,
                channels,
                points,
            })
            .unwrap(),
            &fixture.deform.output,
        );
    }

    #[test]
    fn perception_view_pipeline_matches_oracle_words() {
        let fixture: PerceptionViewFixture = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/qwen_drive/perception-view.json"
        )))
        .unwrap();
        assert_bits(
            "layer norm",
            layer_norm_2d(
                &Tensor4::new(
                    values(&fixture.layer_norm.input.values),
                    fixture.layer_norm.input.shape,
                )
                .unwrap(),
                &values(&fixture.layer_norm.weight),
                &values(&fixture.layer_norm.bias),
                f32::from_bits(fixture.layer_norm.epsilon),
            )
            .unwrap()
            .values(),
            &fixture.layer_norm.output,
        );
        let transposed = conv_transpose2d(
            &Tensor4::new(
                values(&fixture.transpose.input.values),
                fixture.transpose.input.shape,
            )
            .unwrap(),
            &values(&fixture.transpose.weight),
            fixture.transpose.weight_shape,
            Some(&values(&fixture.transpose.bias)),
            fixture.transpose.stride,
            fixture.transpose.padding,
        )
        .unwrap();
        assert_eq!(transposed.shape(), fixture.transpose.output_shape);
        assert_bits("transpose", transposed.values(), &fixture.transpose.output);
        assert_bits(
            "resize aligned",
            resize_bilinear_aligned(
                &Tensor4::new(
                    values(&fixture.resize_aligned.input.values),
                    fixture.resize_aligned.input.shape,
                )
                .unwrap(),
                fixture.resize_aligned.output_hw[0],
                fixture.resize_aligned.output_hw[1],
                true,
            )
            .unwrap()
            .values(),
            &fixture.resize_aligned.output,
        );
        assert_bits(
            "depth",
            depth_softmax(
                &Tensor4::new(values(&fixture.depth.values), fixture.depth.shape).unwrap(),
            )
            .unwrap()
            .values(),
            &fixture.depth_output,
        );
        let geometry = &fixture.geometry;
        let matrices = |source: &[Vec<u32>]| {
            source
                .iter()
                .map(|matrix| values(matrix).try_into().unwrap())
                .collect::<Vec<[f32; 16]>>()
        };
        let (coords, point_indices) = frustum_voxel_coordinates(&ViewGeometry {
            frustum_range: values(&geometry.frustum_range).try_into().unwrap(),
            frustum_size: values(&geometry.frustum_size).try_into().unwrap(),
            pc_range: values(&geometry.pc_range).try_into().unwrap(),
            voxel_size: values(&geometry.voxel_size).try_into().unwrap(),
            voxel_shape: geometry.voxel_shape,
            lidar2img: &matrices(&geometry.lidar2img),
            lidar2ego: &matrices(&geometry.lidar2ego),
        })
        .unwrap();
        assert_eq!(coords, geometry.coords);
        assert_eq!(point_indices, geometry.point_indices);
        assert_bits(
            "bev",
            &voxel_to_bev_tokens(
                &values(&fixture.bev.values),
                fixture.bev.shape,
                &values(&fixture.bev.weight),
                fixture.bev.weight_shape,
                &values(&fixture.bev.bias),
            )
            .unwrap(),
            &fixture.bev.output,
        );
    }

    #[test]
    #[ignore = "requires RMI_QWEN_DRIVE_PERCEPTION"]
    fn qwen_drive_perception_loads_released_view_backbone() {
        let source = crate::core::loader::GGUFLoader::from_file(
            std::env::var_os("RMI_QWEN_DRIVE_PERCEPTION").unwrap(),
        )
        .unwrap();
        let model = ViewBackbone::from_source(&source).unwrap();
        assert_eq!(model.config().image_size, [896, 512]);
        let _ = BevFormer::from_source(&source).unwrap();
        let _ = PerceptionHeads::from_source(&source).unwrap();
        let model = QwenDrivePerception::from_source(&source).unwrap();
        assert_eq!(model.config().bev, [200, 200]);
    }

    #[test]
    #[ignore = "requires RMI_QWEN_DRIVE_PLANNER_SFT"]
    fn qwen_drive_planner_loads_all_released_weights() {
        let source = crate::core::loader::GGUFLoader::from_file(
            std::env::var_os("RMI_QWEN_DRIVE_PLANNER_SFT").unwrap(),
        )
        .unwrap();
        let expert = PlanningExpert::from_source(&source).unwrap();
        assert_eq!(
            (expert.config().layers, expert.config().future_points),
            (32, 50)
        );
    }

    fn planner_request() -> (
        PlannerConfig,
        Vec<crate::models::qwen35::Qwen35DenseKvSnapshot>,
        PlanningScene,
    ) {
        let config = PlannerConfig::from_source(&planner_source()).unwrap();
        let caches = (0..8)
            .map(|index| crate::models::qwen35::Qwen35DenseKvSnapshot {
                layer: index * 4 + 3,
                tokens: 1,
                kv_heads: 4,
                head_dim: 256,
                key: vec![0.0; 1024],
                value: vec![0.0; 1024],
            })
            .collect();
        let scene = PlanningScene {
            token: "fixture".into(),
            views: Vec::new(),
            instruction: "plan".into(),
            history: vec![[0.0; 3]; 16],
            history_velocity: vec![[0.0; 2]; 16],
            history_acceleration: vec![[0.0; 2]; 16],
            nav_command: 0,
            ego_status: [0.0; 8],
        };
        (config, caches, scene)
    }

    #[test]
    fn planner_request_validation_rejects_invalid_inputs() {
        let (config, caches, scene) = planner_request();
        validate_sample_request(&config, &caches, &scene, 1, 10).unwrap();

        assert!(validate_sample_request(&config, &caches[..7], &scene, 1, 10).is_err());
        let mut bad_shape = caches.clone();
        bad_shape[0].head_dim = 128;
        assert!(validate_sample_request(&config, &bad_shape, &scene, 1, 10).is_err());
        assert!(validate_sample_request(&config, &caches, &scene, 1, 0).is_err());
        let mut bad_nav = scene.clone();
        bad_nav.nav_command = 3;
        assert!(validate_sample_request(&config, &caches, &bad_nav, 1, 10).is_err());
        let mut non_finite = scene;
        non_finite.history[0][0] = f32::NAN;
        assert!(validate_sample_request(&config, &caches, &non_finite, 1, 10).is_err());
    }
}
