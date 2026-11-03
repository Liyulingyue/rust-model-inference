use super::*;
use crate::core::tensor::TensorInfo;

struct Source {
    meta: HashMap<String, MetaValue>,
    tensors: HashMap<String, (TensorInfo, Vec<u8>)>,
}

impl TensorSource for Source {
    fn metadata(&self, key: &str) -> Option<&MetaValue> {
        self.meta.get(key)
    }
    fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name).map(|t| &t.0)
    }
    fn tensor_slice(&self, name: &str) -> Option<&[u8]> {
        self.tensors.get(name).map(|t| t.1.as_slice())
    }
}

fn source() -> Source {
    let contract: serde_json::Value = serde_json::from_str(CONTRACT).unwrap();
    let meta = [
        ("general.architecture", "pivo".to_string()),
        ("pivo.format", "pivo-f32-v1".into()),
        ("pivo.config", contract["config"].to_string()),
        ("pivo.lexicon", "hello HH AH0 L OW1\n".into()),
    ]
    .into_iter()
    .map(|(k, v)| (k.into(), MetaValue::String(v)))
    .collect();
    let tensors = contract["tensors"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(name, dims)| {
            let shape: Vec<u64> = serde_json::from_value(dims.clone()).unwrap();
            let count = shape.iter().product::<u64>() as usize;
            (
                name.clone(),
                (
                    TensorInfo {
                        name: name.clone(),
                        dims: shape.into_iter().rev().collect(),
                        ggml_type: GGMLType::F32,
                        offset: 0,
                    },
                    vec![0; count * 4],
                ),
            )
        })
        .collect();
    Source { meta, tensors }
}

#[test]
fn loader_rejects_arch_config_missing_component_shape_dtype_length_and_nan() {
    let mut s = source();
    assert!(Model::from_source(&s, ComputePolicy::Cpu).is_ok());
    assert!(Model::from_source(&s, ComputePolicy::Auto).is_ok());
    assert!(Model::from_source(&s, ComputePolicy::Vulkan)
        .err()
        .unwrap()
        .contains("no Vulkan"));
    for (key, value) in [
        ("general.architecture", "qwen3"),
        ("pivo.format", "unknown"),
        ("pivo.config", "{}"),
    ] {
        let old = s
            .meta
            .insert(key.into(), MetaValue::String(value.into()))
            .unwrap();
        assert!(Model::from_source(&s, ComputePolicy::Cpu).is_err());
        s.meta.insert(key.into(), old);
    }
    let name = "vocoder.step.signal_out.weight";
    let tensor = s.tensors.remove(name).unwrap();
    assert!(Model::from_source(&s, ComputePolicy::Cpu)
        .err()
        .unwrap()
        .contains(name));
    s.tensors.insert(name.into(), tensor);
    s.tensors.get_mut(name).unwrap().0.dims.swap(0, 1);
    assert!(Model::from_source(&s, ComputePolicy::Cpu).is_err());
    s.tensors.get_mut(name).unwrap().0.dims.swap(0, 1);
    s.tensors.get_mut(name).unwrap().0.ggml_type = GGMLType::F16;
    assert!(Model::from_source(&s, ComputePolicy::Cpu).is_err());
    s.tensors.get_mut(name).unwrap().0.ggml_type = GGMLType::F32;
    s.tensors.get_mut(name).unwrap().1[0..4].copy_from_slice(&f32::NAN.to_le_bytes());
    assert!(Model::from_source(&s, ComputePolicy::Cpu).is_err());
    s.tensors.get_mut(name).unwrap().1.clear();
    assert!(Model::from_source(&s, ComputePolicy::Cpu).is_err());
}

#[test]
fn frontend_normalizes_and_preserves_specials_and_first_pronunciation() {
    let front = text::Frontend::new("hello HH AH0 L OW1\nhello(2) HH EH1 L OW0\nworld W ER1 L D\n")
        .unwrap();
    let actual = front
        .phonemize("“Ｈｅｌｌｏ, world!”", &HashMap::new())
        .unwrap();
    assert_eq!(actual.normalized, "Hello, world!");
    assert_eq!(
        actual.tokens,
        vec![1, 16, 40, 19, 68, 5, 3, 30, 56, 19, 12, 7, 2]
    );
    for input in ["", ".?!", "zzvo", "1/2", "2025-02-29", "hello🙂"] {
        assert!(front.phonemize(input, &HashMap::new()).is_err(), "{input}");
    }
    let overrides = HashMap::from([(
        "ZZVO".into(),
        vec!["Z".into(), "IY1".into(), "V".into(), "OW0".into()],
    )]);
    assert_eq!(
        front.phonemize("Zzvo", &overrides).unwrap().tokens,
        vec![1, 32, 65, 29, 67, 2]
    );
    let conflicting = HashMap::from([
        ("hello".into(), vec!["HH".into()]),
        ("HELLO".into(), vec!["B".into()]),
    ]);
    assert!(front
        .phonemize("hello", &conflicting)
        .err()
        .unwrap()
        .contains("Duplicate"));
}

#[test]
fn overflow_in_finite_weights_returns_an_error_before_pitch_indexing() {
    let mut s = source();
    for (name, value) in [
        ("acoustic.phase_projection.weight", 1f32),
        ("acoustic.feature_norm.weight", 1f32),
        ("acoustic.feature_head.weight", f32::MAX),
    ] {
        for bytes in s.tensors.get_mut(name).unwrap().1.chunks_exact_mut(4) {
            bytes.copy_from_slice(&value.to_le_bytes());
        }
    }
    let model = Model::from_source(&s, ComputePolicy::Cpu).unwrap();
    let request = SpeechRequest {
        input: "hello".into(),
        voice: "Pi".into(),
        speed: 1.,
        overrides: HashMap::new(),
    };
    assert!(matches!(
        model.synthesize(&request),
        Err(SynthesisError::Numerical(_))
    ));
}

#[test]
fn normalization_covers_numbers_currency_dates_initialisms_and_punctuation() {
    for (input, expected) in [
        ("NASA & CPU + 21st", "nasa and C P U plus twenty first"),
        ("Dr. Smith paid £1.01 and $2.50.", "Doctor Smith paid one pound and one penny and two dollars and fifty cents."),
        ("2024-02-29: -3.05, 1,234 and 20%.", "February twenty ninth, two thousand twenty four: minus three point zero five, one thousand two hundred thirty four and twenty percent ."),
        ("Hello—world…", "Hello,world..."),
        ("١٢ and २१st cost $٣.٥٠.", "twelve and twenty first cost three dollars and fifty cents."),
    ] {
        assert_eq!(text::normalize(input).unwrap(), expected);
    }
    assert!(text::normalize("$1.234").is_err());
}

#[test]
fn duration_rounding_phase_and_budgets_match_reference_rules() {
    assert_eq!(
        acoustic::durations(&[1, 10, 3, 10, 2], &[100., 0., 0., 2f32.ln(), 100.], 2.).unwrap(),
        vec![0, 1, 0, 1, 0]
    );
    assert!(acoustic::durations(&[10; 16], &[100.; 16], 1.).is_err());
    assert!(acoustic::durations(&[10], &[f32::NAN], 1.).is_err());
    let result = acoustic::expand(&[1.; 96], &[2]);
    assert_eq!(result.len(), 200);
    assert_eq!(result[96], -0.5);
    assert_eq!(result[196], 0.5);
    assert!((result[97] - 0.5f32.sqrt()).abs() < 1e-6);
}

fn flatten(value: &serde_json::Value) -> Vec<f32> {
    match value {
        serde_json::Value::Array(values) => values.iter().flat_map(flatten).collect(),
        _ => vec![value.as_f64().unwrap() as f32],
    }
}

fn compare(name: &str, actual: &[f32], expected: &[f32], max: f64, rms: f64) {
    assert_eq!(actual.len(), expected.len(), "{name} shape");
    assert!(actual.iter().all(|v| v.is_finite()), "{name} non-finite");
    let max_abs = actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (f64::from(*a) - f64::from(*b)).abs())
        .fold(0f64, f64::max);
    let rmse = (actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (f64::from(*a) - f64::from(*b)).powi(2))
        .sum::<f64>()
        / actual.len() as f64)
        .sqrt();
    println!(
        "{name}: count={} max_abs={max_abs:.9e} rmse={rmse:.9e}",
        actual.len()
    );
    assert!(
        max_abs <= max && rmse <= rms,
        "{name}: max_abs={max_abs} rmse={rmse}"
    );
}

#[test]
#[ignore = "requires pinned PiVo GGUF and official ONNX fixtures; see tools/oracle/pivo"]
fn official_onnx_checkpoints_and_waveform() {
    let source =
        crate::core::loader::GGUFLoader::from_file(std::env::var("PIVO_GGUF").unwrap()).unwrap();
    let policy = if std::env::var_os("PIVO_AUTO").is_some() {
        ComputePolicy::Auto
    } else {
        ComputePolicy::Cpu
    };
    let model = Model::from_source(&source, policy).unwrap();
    let cases: Vec<serde_json::Value> =
        serde_json::from_slice(&std::fs::read(std::env::var("PIVO_FIXTURES").unwrap()).unwrap())
            .unwrap();
    for case in cases {
        let request: SpeechRequest = serde_json::from_value(case["request"].clone()).unwrap();
        println!("case: {:?}", case["request"]);
        let front = model
            .frontend
            .phonemize(&request.input, &request.overrides)
            .unwrap();
        assert_eq!(front.normalized, case["normalized"].as_str().unwrap());
        assert_eq!(
            front.tokens,
            serde_json::from_value::<Vec<usize>>(case["tokens"].clone()).unwrap()
        );
        let (encoded, logs) = model.encode(&front.tokens, usize::from(request.voice == "Vo"));
        compare("encoded", &encoded, &flatten(&case["encoded"]), 1e-4, 1e-5);
        compare(
            "log_durations",
            &logs,
            &flatten(&case["log_durations"]),
            1e-4,
            1e-5,
        );
        let durations = acoustic::durations(&front.tokens, &logs, request.speed).unwrap();
        assert_eq!(
            durations,
            serde_json::from_value::<Vec<usize>>(case["durations"].clone()).unwrap()
        );
        let expanded = acoustic::expand(&flatten(&case["encoded"]), &durations);
        compare(
            "expansion_phase",
            &expanded,
            &flatten(&case["expanded"]),
            1e-6,
            1e-7,
        );
        compare(
            "decoder",
            &model.decode(&expanded),
            &flatten(&case["features"]),
            1e-4,
            1e-5,
        );
        for record in case["conditioning"].as_array().unwrap() {
            compare(
                "conditioning",
                &model.conditioning(&flatten(&record["inputs"]["features"])),
                &flatten(&record["outputs"][0]),
                2e-5,
                2e-6,
            );
        }
        for record in case["steps"].as_array().unwrap() {
            let input = &record["inputs"];
            let mut state = vocoder::State {
                memory: flatten(&input["memory"]),
                gru: [
                    flatten(&input["gru1"]),
                    flatten(&input["gru2"]),
                    flatten(&input["gru3"]),
                ],
                frame: flatten(&input["frame_state"]),
            };
            let period = flatten(&input["period"])[0] as usize;
            compare(
                "excitation",
                &model.vocoder_step(&flatten(&input["conditioning"]), period, &mut state),
                &flatten(&record["outputs"][0]),
                2e-5,
                2e-6,
            );
            for (actual, expected) in [
                &state.memory,
                &state.gru[0],
                &state.gru[1],
                &state.gru[2],
                &state.frame,
            ]
            .iter()
            .zip(record["outputs"].as_array().unwrap().iter().skip(1))
            {
                compare("vocoder_state", actual, &flatten(expected), 2e-5, 2e-6);
            }
        }
        let reference_features = flatten(&case["features"]);
        let vocoder_pcm = model.vocode(&reference_features);
        compare(
            "vocoder_pcm",
            &vocoder_pcm,
            &flatten(&case["pcm"]),
            5e-3,
            7.5e-4,
        );
        let native_features = model.decode(&acoustic::expand(&encoded, &durations));
        compare(
            "acoustic_features",
            &native_features,
            &reference_features,
            1e-4,
            1e-5,
        );
        let native_periods: Vec<_> = native_features
            .chunks_exact(20)
            .map(vocoder::period)
            .collect();
        assert_eq!(
            native_periods,
            serde_json::from_value::<Vec<usize>>(case["periods"].clone()).unwrap()
        );
        let pcm = model.synthesize(&request).unwrap();
        // FARGAN feedback amplifies even one-ULP perturbations in the official graph.
        compare("pcm", &pcm, &flatten(&case["pcm"]), 5e-3, 7.5e-4);
    }
}
