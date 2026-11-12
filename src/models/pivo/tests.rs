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
    // The strict policy needs a device: a build without the Vulkan feature
    // has none, and a build with one still needs a device to be present.
    if let Err(error) = Model::from_source(&s, ComputePolicy::Vulkan) {
        assert!(error.contains("Vulkan"), "{error}");
    }
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
fn loader_accepts_q8_0_weight_matrices_and_dequantizes_them() {
    let mut s = source();
    let name = "vocoder.step.signal_out.weight";
    let count = 40 * 128;
    // One Q8_0 block: an F16 scale followed by 32 int8 values. A scale of
    // 1.0 is exact in F16, so the dequantized value is the payload value.
    let payload: Vec<i8> = (0..32).map(|i| i as i8 - 16).collect();
    let mut block = vec![0x00, 0x3C];
    block.extend(payload.iter().map(|value| *value as u8));
    let mut data = Vec::new();
    for _ in 0..count / 32 {
        data.extend_from_slice(&block);
    }
    s.tensors.get_mut(name).unwrap().0.ggml_type = GGMLType::Q8_0;
    s.tensors.get_mut(name).unwrap().1 = data;
    s.meta.insert(
        "pivo.format".into(),
        MetaValue::String("pivo-q8_0-v1".into()),
    );
    let model = Model::from_source(&s, ComputePolicy::Cpu).unwrap();
    assert_eq!(model.format, WeightFormat::Q8_0);
    let tensor = &model.weights[name];
    assert_eq!(tensor.data.len(), count);
    for (index, value) in tensor.data.iter().enumerate() {
        assert_eq!(*value, payload[index % 32] as f32, "element {index}");
    }
}

#[test]
fn loader_rejects_q8_0_where_a_block_does_not_fit() {
    // A one-dimensional tensor has no block-aligned element count and
    // holds a norm, a bias or a statistic, so it stays F32.
    let mut s = source();
    s.tensors
        .get_mut("acoustic.feature_mean")
        .unwrap()
        .0
        .ggml_type = GGMLType::Q8_0;
    s.tensors.get_mut("acoustic.feature_mean").unwrap().1 = vec![0; 34];
    assert!(Model::from_source(&s, ComputePolicy::Cpu).is_err());

    // 80 elements is not a whole number of 32-value blocks.
    let mut s = source();
    let name = "vocoder.step.cond_gain.weight";
    s.tensors.get_mut(name).unwrap().0.ggml_type = GGMLType::Q8_0;
    s.tensors.get_mut(name).unwrap().1 = vec![0; 34 * 3];
    assert!(Model::from_source(&s, ComputePolicy::Cpu).is_err());

    // A truncated payload.
    let mut s = source();
    let name = "vocoder.step.signal_out.weight";
    s.tensors.get_mut(name).unwrap().0.ggml_type = GGMLType::Q8_0;
    s.tensors.get_mut(name).unwrap().1 = vec![0; 34 * 10];
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

/// Per-checkpoint error budgets against the official ONNX graph.
///
/// The F32 column is the implementation error: the Rust graph and the
/// official one round differently, so the numbers sit far below the
/// tolerance the vocoder's feedback loop can amplify. The Q8_0 column adds
/// the weight quantization loss on top: the export stores every 32-value
/// block with a single F16 scale, so each weight carries a relative error of
/// about 1/127 and the vocoder compounds it.
///
/// Both columns are measured, not guessed; see
/// tools/oracle/pivo/validation.md for the runs they come from.
struct Gates {
    encoder: (f64, f64),
    log_durations: (f64, f64),
    expansion: (f64, f64),
    decoder: (f64, f64),
    conditioning: (f64, f64),
    excitation: (f64, f64),
    state: (f64, f64),
    vocoder_pcm: (f64, f64),
    features: (f64, f64),
    pcm: (f64, f64),
}

impl Gates {
    fn for_format(format: WeightFormat) -> Self {
        match format {
            WeightFormat::F32 => Self {
                encoder: (1e-4, 1e-5),
                log_durations: (1e-4, 1e-5),
                expansion: (1e-6, 1e-7),
                decoder: (1e-4, 1e-5),
                conditioning: (2e-5, 2e-6),
                excitation: (2e-5, 2e-6),
                state: (2e-5, 2e-6),
                vocoder_pcm: (5e-3, 7.5e-4),
                features: (1e-4, 1e-5),
                pcm: (5e-3, 7.5e-4),
            },
            // The Q8_0 column is the measured degradation of the quantized
            // export against the official ONNX graph, with margin. These
            // are regression guards, not a quality claim: the format is not
            // usable, and the numbers behind that are in
            // tools/oracle/pivo/validation.md.
            WeightFormat::Q8_0 => Self {
                encoder: (6e-2, 1.2e-2),
                log_durations: (7e-2, 2.5e-2),
                expansion: (1e-6, 1e-7),
                decoder: (3.5e-1, 2.5e-2),
                conditioning: (1.2e-1, 1.5e-2),
                excitation: (2.5e-3, 1e-3),
                state: (3.5e-2, 5e-3),
                vocoder_pcm: (1.2, 1.2e-1),
                features: (7e-1, 3.5e-2),
                pcm: (1.6, 1.6e-1),
            },
        }
    }
}

/// The measured worst mismatch rate of the discrete contract for the Q8_0
/// export: 7.14% of the frame durations and 23.23% of the pitch periods
/// across the eight oracle cases. The bands below hold that measurement
/// with margin so a further degradation is caught.
const Q8_0_DURATION_MISMATCH: f64 = 0.10;
const Q8_0_PERIOD_MISMATCH: f64 = 0.25;

/// The discrete contract: the frame durations and the pitch periods.
///
/// The F32 export reproduces both exactly. The Q8_0 export moves every
/// weight matrix onto the 1/127 grid of its per-block scale, and both
/// decisions flip: the FARGAN feedback turns a moved weight into a
/// different frame count and a different period. The mismatch rate is
/// measured and bounded instead of being asserted away, because the format
/// is not usable; the numbers behind that statement are in
/// tools/oracle/pivo/validation.md.
fn compare_discrete(label: &str, actual: &[usize], expected: &[usize], format: WeightFormat) {
    match format {
        WeightFormat::F32 => assert_eq!(actual, expected, "{label}"),
        WeightFormat::Q8_0 => {
            assert_eq!(actual.len(), expected.len(), "{label} shape");
            let mismatched = actual
                .iter()
                .zip(expected.iter())
                .filter(|(actual, expected)| actual != expected)
                .count();
            let rate = mismatched as f64 / expected.len().max(1) as f64;
            println!(
                "{label} mismatch: {mismatched}/{} ({:.2}%)",
                expected.len(),
                rate * 100.0
            );
            let band = if label == "durations" {
                Q8_0_DURATION_MISMATCH
            } else {
                Q8_0_PERIOD_MISMATCH
            };
            assert!(
                rate <= band,
                "{label} mismatch rate {rate} exceeds the measured {band}"
            );
        }
    }
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
    let gates = Gates::for_format(model.format);
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
        compare(
            "encoded",
            &encoded,
            &flatten(&case["encoded"]),
            gates.encoder.0,
            gates.encoder.1,
        );
        compare(
            "log_durations",
            &logs,
            &flatten(&case["log_durations"]),
            gates.log_durations.0,
            gates.log_durations.1,
        );
        let durations = acoustic::durations(&front.tokens, &logs, request.speed).unwrap();
        let official_durations: Vec<usize> =
            serde_json::from_value(case["durations"].clone()).unwrap();
        compare_discrete("durations", &durations, &official_durations, model.format);
        // The pipeline stages run on the official frame timing so the shapes
        // stay comparable when the quantized export changes a frame count.
        // The frame timing itself is checked above.
        let expanded = acoustic::expand(&flatten(&case["encoded"]), &official_durations);
        compare(
            "expansion_phase",
            &expanded,
            &flatten(&case["expanded"]),
            gates.expansion.0,
            gates.expansion.1,
        );
        compare(
            "decoder",
            &model.decode(&expanded),
            &flatten(&case["features"]),
            gates.decoder.0,
            gates.decoder.1,
        );
        for record in case["conditioning"].as_array().unwrap() {
            compare(
                "conditioning",
                &model.conditioning(&flatten(&record["inputs"]["features"])),
                &flatten(&record["outputs"][0]),
                gates.conditioning.0,
                gates.conditioning.1,
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
                gates.excitation.0,
                gates.excitation.1,
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
                compare(
                    "vocoder_state",
                    actual,
                    &flatten(expected),
                    gates.state.0,
                    gates.state.1,
                );
            }
        }
        let reference_features = flatten(&case["features"]);
        let vocoder_pcm = model.vocode(&reference_features);
        compare(
            "vocoder_pcm",
            &vocoder_pcm,
            &flatten(&case["pcm"]),
            gates.vocoder_pcm.0,
            gates.vocoder_pcm.1,
        );
        let native_features = model.decode(&acoustic::expand(&encoded, &official_durations));
        compare(
            "acoustic_features",
            &native_features,
            &reference_features,
            gates.features.0,
            gates.features.1,
        );
        let native_periods: Vec<_> = native_features
            .chunks_exact(20)
            .map(vocoder::period)
            .collect();
        let reference_periods =
            serde_json::from_value::<Vec<usize>>(case["periods"].clone()).unwrap();
        compare_discrete(
            "pitch_periods",
            &native_periods,
            &reference_periods,
            model.format,
        );
        // The end-to-end waveform runs on the official frame timing for the
        // quantized export, because its own frame count can differ; the
        // F32 export goes through the real entry point.
        let pcm = match model.format {
            WeightFormat::F32 => model.synthesize(&request).unwrap(),
            WeightFormat::Q8_0 => model.vocode(&native_features),
        };
        // FARGAN feedback amplifies even one-ULP perturbations in the official graph.
        compare(
            "pcm",
            &pcm,
            &flatten(&case["pcm"]),
            gates.pcm.0,
            gates.pcm.1,
        );
    }
}

/// The device encoder has to keep the host's frame durations.
///
/// The durations are rounded to whole frames and the rounding margin of the
/// shortest oracle case is 0.0024 frames, so the device and the host have to
/// agree on the log duration to far better than that before the frame count
/// can differ. The frames decide the length of the waveform, so a difference
/// here is an audible difference and not a rounding detail.
#[cfg(feature = "vulkan")]
#[test]
fn device_encoder_keeps_the_host_frame_durations() {
    let path = std::path::Path::new("models/PiVo-1.4M/pivo-F32.gguf");
    if !path.exists() {
        eprintln!("the PiVo model file is missing, the device check is skipped");
        return;
    }
    // The strict policy brings the device up and keeps it in the legacy
    // scope for the rest of the test.
    let _scope = match ComputePolicy::Vulkan.configure_cli() {
        Ok(scope) => scope,
        Err(error) => {
            eprintln!("no Vulkan device, the device check is skipped: {error}");
            return;
        }
    };
    let source = crate::core::loader::GGUFLoader::from_file(path).unwrap();
    let host = Model::from_source(&source, ComputePolicy::Cpu).unwrap();
    let device = Model::from_source(&source, ComputePolicy::Vulkan).unwrap();
    let text = "NASA is ready for the next mission.";
    let front = host.frontend.phonemize(text, &HashMap::new()).unwrap();
    let (host_encoded, host_logs) = host.encode(&front.tokens, 0);
    let device_encoded = device.encode_on_device(&front.tokens, 0).unwrap();
    let device_logs = device.duration_head(&device_encoded);

    let scale = host_encoded.iter().fold(0f32, |m, v| m.max(v.abs()));
    let error = host_encoded
        .iter()
        .zip(&device_encoded)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let log_error = host_logs
        .iter()
        .zip(&device_logs)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!(
        "encoder: |x|max={scale:e} error={error:e} relative={:e} log-duration error={log_error:e}",
        error / scale
    );
    let host_frames = acoustic::durations(&front.tokens, &host_logs, 1.0).unwrap();
    let device_frames = acoustic::durations(&front.tokens, &device_logs, 1.0).unwrap();
    eprintln!("frames: host={host_frames:?}");
    eprintln!("frames: device={device_frames:?}");
    assert_eq!(
        host_frames, device_frames,
        "the device and the host disagree on the frame durations"
    );
}
