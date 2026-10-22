use rust_model_inference::core::thread_pool::ComputePool;
use rust_model_inference::models::diffusion::dreamx::kernels::{
    conv3d_with_options, load_float_values, Conv3dSpec, Linear,
};
use rust_model_inference::{GGUFLoader, TensorSource};
use serde_json::Value;
use std::path::Path;

fn shape<const N: usize>(value: &Value) -> [usize; N] {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|n| n.as_u64().unwrap() as usize)
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
}

fn values(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap();
    assert!(!bytes.is_empty() && bytes.len().is_multiple_of(4));
    bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect()
}

#[test]
#[ignore = "requires real Astronex GGUFs and independent checkpoint_operators.py fixtures"]
fn astronex_checkpoint_exact_operators_match_independent_scalar_bits() {
    assert!(
        rust_model_inference::ops::scalar_mode(),
        "run with RMI_SCALAR=1"
    );
    let root = Path::new("artifacts/astronex-checkpoint-operators");
    let fixture: Value =
        serde_json::from_slice(&std::fs::read(root.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(fixture["schema_version"], 1);
    let main = GGUFLoader::from_file("models/Astronex-World/Astronex-World-BF16.gguf").unwrap();
    let components =
        GGUFLoader::from_file("models/Astronex-World/mmproj-Astronex-World.gguf").unwrap();
    let config =
        rust_model_inference::models::diffusion::astronex_world::AstronexConfig::from_sources(
            &main,
            &components,
        )
        .unwrap();
    assert_eq!(fixture["official_commit"], config.oracle_commit);
    assert_eq!(fixture["contract_commit"], config.source_revision);
    let pool = ComputePool::new(1);
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 12);
    for case in cases {
        let source: &dyn TensorSource = match case["role"].as_str().unwrap() {
            "main" => &main,
            "components" => &components,
            other => panic!("unexpected role: {other}"),
        };
        let name = case["name"].as_str().unwrap();
        let input = values(&root.join(case["input"].as_str().unwrap()));
        let bias_name = case["bias"].as_str();
        let actual = match case["kind"].as_str().unwrap() {
            "linear" => {
                let [rows, n_in] = shape::<2>(&case["input_shape"]);
                let [n_out, weight_in] = shape::<2>(&case["weight_shape"]);
                assert_eq!(weight_in, n_in);
                assert_eq!(shape::<2>(&case["output_shape"]), [rows, n_out]);
                Linear::from_source(source, &format!("{name}.weight"), bias_name, n_in, n_out)
                    .unwrap()
                    .forward(&pool, &input, rows)
                    .unwrap()
            }
            "conv3d" => {
                let input_shape = shape::<4>(&case["input_shape"]);
                let weight_shape = shape::<5>(&case["weight_shape"]);
                let gguf_shape: Vec<u64> = weight_shape.iter().rev().map(|&n| n as u64).collect();
                let weight =
                    load_float_values(source, &format!("{name}.weight"), &gguf_shape).unwrap();
                let bias = load_float_values(source, bias_name.unwrap(), &[weight_shape[0] as u64])
                    .unwrap();
                let (out, output_shape) = conv3d_with_options(
                    &pool,
                    &input,
                    input_shape,
                    &weight,
                    weight_shape,
                    Some(&bias),
                    Conv3dSpec {
                        stride: shape::<3>(&case["stride"]),
                        ..Conv3dSpec::default()
                    },
                )
                .unwrap();
                assert_eq!(output_shape, shape::<4>(&case["output_shape"]));
                out
            }
            "embedding" => {
                let embedding =
                    Linear::from_source(source, &format!("{name}.weight"), None, 4096, 256384)
                        .unwrap();
                let ids = case["token_ids"].as_array().unwrap();
                assert_eq!(shape::<2>(&case["output_shape"]), [ids.len(), 4096]);
                let mut output = vec![0.0; ids.len() * 4096];
                for (id, row) in ids.iter().zip(output.chunks_exact_mut(4096)) {
                    embedding
                        .embedding_lookup(id.as_u64().unwrap() as u32, row)
                        .unwrap();
                }
                output
            }
            other => panic!("unexpected operator: {other}"),
        };
        let expected = values(&root.join(case["expected"].as_str().unwrap()));
        assert_eq!(actual.len(), expected.len(), "{name}");
        for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "{name} first divergence at {index}: actual={actual:e}, oracle={expected:e}"
            );
        }
        eprintln!(
            "{name}: {} real-weight F32 values match raw bits",
            actual.len()
        );
    }
}
