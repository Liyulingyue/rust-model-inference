use super::*;
#[test]
fn loader_rejects_invalid_contract_shape_type_and_payload() {
    use crate::core::tensor::{GGMLType, MetaValue, TensorInfo};
    struct Source {
        architecture: MetaValue,
        contract: MetaValue,
        info: Option<TensorInfo>,
    }
    impl TensorSource for Source {
        fn metadata(&self, key: &str) -> Option<&MetaValue> {
            match key {
                "general.architecture" => Some(&self.architecture),
                "pjit.contract" => Some(&self.contract),
                _ => None,
            }
        }
        fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
            self.info.as_ref().filter(|info| info.name == name)
        }
        fn tensor_slice(&self, _: &str) -> Option<&[u8]> {
            Some(&[])
        }
    }
    let source = |dims, kind, contract| {
        Arc::new(Source {
            architecture: MetaValue::String("pjit".into()),
            contract: MetaValue::Uint32(contract),
            info: Some(TensorInfo {
                name: "patch_embed.proj1.weight".into(),
                dims,
                ggml_type: kind,
                offset: 0,
            }),
        })
    };
    for (dims, kind, contract, message) in [
        (vec![3072, 1024], GGMLType::F32, 2, "contract 1"),
        (vec![32, 1024], GGMLType::F32, 1, "Invalid tensor"),
        (vec![3072, 1024], GGMLType::I8, 1, "Invalid tensor"),
        (vec![3072, 1024], GGMLType::Q8_0, 1, "data length"),
    ] {
        let error = PjitDit::load(source(dims, kind, contract), 1)
            .err()
            .unwrap();
        assert!(error.contains(message), "{error}");
    }
}
#[test]
fn parallel_attention_preserves_head_and_token_order() {
    let input: Vec<_> = (0..3 * 23 * 128)
        .map(|i| ((i % 31) as f32 - 15.) / 11.)
        .collect();
    let spec = crate::ops::nn::AttentionSpec {
        query_tokens: 3,
        key_tokens: 3,
        query_heads: 23,
        key_value_heads: 23,
        head_dim: 128,
        causal: false,
        scale: 1. / 128f32.sqrt(),
    };
    let expected = crate::ops::nn::attention_on_cpu(&input, &input, &input, spec).unwrap();
    for threads in [1, 4] {
        assert_eq!(
            dit::cpu_attention(&input, &input, &input, spec, &ComputePool::new(threads)).unwrap(),
            expected
        );
    }
}
#[test]
#[ignore = "requires RMI_PJIT_TEXT and RMI_PJIT_TOKENS JSON fixtures"]
fn pjit_released_tokenizer_matches_official() {
    let source = crate::format::ggufrs::open_model_source(
        std::path::Path::new(&std::env::var("RMI_PJIT_TEXT").unwrap()),
        crate::format::ggufrs::ComponentRole::Llm,
    )
    .unwrap();
    let cases: Vec<serde_json::Value> =
        serde_json::from_slice(&std::fs::read(std::env::var("RMI_PJIT_TOKENS").unwrap()).unwrap())
            .unwrap();
    for case in cases {
        let expected: Vec<u32> = serde_json::from_value(case["ids"].clone()).unwrap();
        assert_eq!(
            caption_ids(source.as_ref(), case["prompt"].as_str().unwrap()).unwrap(),
            expected
        );
    }
}
#[test]
fn sampling_timesteps_match_pinned_torch_linspace() {
    let expected = [
        1000, 980, 960, 940, 920, 900, 880, 860, 840, 820, 800, 780, 760, 740, 720, 700, 680, 660,
        640, 620, 600, 580, 560, 540, 520, 500, 480, 459, 440, 420, 399, 380, 359, 340, 320, 299,
        280, 260, 240, 220, 199, 179, 160, 140, 120, 99, 80, 60, 40, 20,
    ];
    assert_eq!(
        (0..50)
            .map(|i| (sampling_time(i, 50) * 1000.) as usize)
            .collect::<Vec<_>>(),
        expected
    );
    assert_eq!(sampling_time(50, 50), 0.);
}
#[test]
fn pixel_patch_order_and_crop_round_trip() {
    let pixels: Vec<_> = (0..3 * 33 * 35).map(|i| i as f32).collect();
    let patches = patchify(&pixels, 35, 33);
    let mut interleaved = vec![0.0; 4 * 3072];
    for p in 0..4 {
        for c in 0..3 {
            for i in 0..1024 {
                interleaved[p * 3072 + i * 3 + c] = patches[p * 3072 + c * 1024 + i];
            }
        }
    }
    assert_eq!(unpatchify(&interleaved, 35, 33), pixels);
}
#[test]
fn options_reject_unbounded_shapes_and_nonfinite_guidance() {
    let mut options = PjitOptions::default();
    options.width = usize::MAX;
    assert!(options.validate().is_err());
    options = PjitOptions::default();
    options.guidance = f32::NAN;
    assert!(options.validate().is_err());
    options = PjitOptions::default();
    options.steps = 0;
    assert!(options.validate().is_err());
}
#[test]
fn apg_is_orthogonal_and_clips_norm() {
    let cond = [1., 0., 0.];
    let other = [1., -100., 0.];
    let mut momentum = vec![0.; 3];
    let guided = apg(&cond, &other, &mut momentum, 2.0);
    assert_eq!(momentum, vec![0., 100., 0.]);
    assert!((guided[0] - 1.).abs() < 1e-6);
    assert!((guided[1] - 20.).abs() < 1e-5);
    assert!(apg(&[0.; 3], &[0.; 3], &mut vec![0.; 3], 15.)
        .iter()
        .all(|x| x.is_finite()));
}
#[test]
fn text_template_disables_thinking() {
    assert_eq!(
        render_caption("猫"),
        "<|im_start|>user\n猫<|im_end|>\n<|im_start|>assistant\n<think>\n"
    );
}
#[test]
#[ignore = "requires RMI_PJIT_GGUF, RMI_PJIT_FIXTURE directory; writes only requested fixture outputs"]
fn pjit_released_dit_matches_fixture() {
    let path = std::env::var("RMI_PJIT_GGUF").unwrap();
    let root = std::path::PathBuf::from(std::env::var("RMI_PJIT_FIXTURE").unwrap());
    let spec: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("input.json")).unwrap()).unwrap();
    let read = |name: &str| {
        std::fs::read(root.join(name))
            .unwrap()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>()
    };
    let source = Arc::from(
        crate::format::ggufrs::open_model_source(
            std::path::Path::new(&path),
            crate::format::ggufrs::ComponentRole::Llm,
        )
        .unwrap(),
    );
    let dit = PjitDit::load(source, 8).unwrap();
    let policy = match std::env::var("RMI_PJIT_COMPUTE").as_deref() {
        Ok("vulkan") => ComputePolicy::Vulkan,
        Ok("auto") => ComputePolicy::Auto,
        _ => ComputePolicy::Cpu,
    };
    let _scope = policy.configure_cli().unwrap();
    let started = std::time::Instant::now();
    let output = dit
        .session()
        .unwrap()
        .predict(
            &read("image.f32"),
            spec["width"].as_u64().unwrap() as usize,
            spec["height"].as_u64().unwrap() as usize,
            &read("text.f32"),
            spec["timestep"].as_u64().unwrap() as usize,
        )
        .unwrap();
    let out = std::env::var("RMI_PJIT_OUTPUT").unwrap();
    std::fs::write(
        out,
        output
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    eprintln!(
        "PJIT_PREDICT_SECONDS={:.6}",
        started.elapsed().as_secs_f64()
    );
}
#[test]
#[ignore = "requires RMI_PJIT_GGUF, RMI_PJIT_TEXT and RMI_PJIT_TEXT_OUTPUT"]
fn pjit_released_caption_features() {
    let open = |var: &str| {
        Arc::from(
            crate::format::ggufrs::open_model_source(
                std::path::Path::new(&std::env::var(var).unwrap()),
                crate::format::ggufrs::ComponentRole::Llm,
            )
            .unwrap(),
        )
    };
    let pipeline = PjitPipeline::load(open("RMI_PJIT_GGUF"), open("RMI_PJIT_TEXT"), 8).unwrap();
    let text = pipeline
        .encode(
            &std::env::var("RMI_PJIT_PROMPT")
                .unwrap_or_else(|_| "A red cube on a white table.".into()),
        )
        .unwrap();
    std::fs::write(
        std::env::var("RMI_PJIT_TEXT_OUTPUT").unwrap(),
        text.iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap();
}
