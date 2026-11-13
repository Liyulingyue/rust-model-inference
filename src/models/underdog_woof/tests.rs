use super::*;

#[test]
fn speech_request_rejects_ignored_or_unbounded_inputs() {
    let mut request: SpeechRequest = serde_json::from_str(r#"{"input":"Hello."}"#).unwrap();
    assert_eq!(request.voice, "juno");
    assert_eq!(request.seed, 0);
    assert!(request.normalize);
    request.validate().unwrap();
    request.voice = "other".into();
    assert!(request.validate().unwrap_err().contains("juno or mira"));
    request.voice = "mira".into();
    request.temperature = f32::NAN;
    assert!(request.validate().is_err());
    request.temperature = 0.0;
    request.input = " ".into();
    assert!(request.validate().is_err());
    request.input = "a".repeat(4097);
    assert!(request.validate().is_err());
    for json in [
        r#"{"input":"Hi","speed":2}"#,
        r#"{"input":"Hi","seed":4294967296}"#,
    ] {
        assert!(serde_json::from_str::<SpeechRequest>(json).is_err());
    }
}

#[test]
fn silence_guard_excludes_leading_and_trailing_silence() {
    assert_eq!(stall(&vec![0.0; 960 * 100]), 0.0);
    let mut signal = vec![0.0; 960 * 25];
    signal.extend(vec![1.0; 960]);
    signal.extend(vec![0.0; 960 * 25]);
    signal.extend(vec![1.0; 960]);
    signal.extend(vec![0.0; 960 * 25]);
    assert_eq!(stall(&signal), 1.0);
}

#[test]
fn rotary_embedding_rotates_halves_with_absolute_positions() {
    let mut x = [1.0, 2.0];
    acoustic::rope(&mut x, 1, 1, &[1.0]);
    assert!((x[0] - (1.0f32.cos() - 2.0 * 1.0f32.sin())).abs() < 1e-6);
    assert!((x[1] - (2.0 * 1.0f32.cos() + 1.0f32.sin())).abs() < 1e-6);
}

struct Header(BTreeMap<String, crate::core::tensor::MetaValue>);
impl TensorSource for Header {
    fn metadata(&self, key: &str) -> Option<&crate::core::tensor::MetaValue> {
        self.0.get(key)
    }
    fn tensor_info(&self, _: &str) -> Option<&crate::core::tensor::TensorInfo> {
        None
    }
    fn tensor_slice(&self, _: &str) -> Option<&[u8]> {
        None
    }
}

#[test]
fn explicit_architecture_config_and_component_errors_are_named() {
    use crate::core::tensor::MetaValue::String as Text;
    let mut header = Header(BTreeMap::from([
        ("general.architecture".into(), Text(ARCHITECTURE.into())),
        (
            "underdog_woof.config".into(),
            Text(contract()["config"].to_string()),
        ),
        ("underdog_woof.format".into(), Text("f32".into())),
    ]));
    assert_eq!(validate_config(&header).unwrap(), "f32");
    assert!(Weights::load(&header)
        .err()
        .unwrap()
        .contains("decoder.inp.bias"));
    header
        .0
        .insert("general.architecture".into(), Text("qwen3".into()));
    assert!(validate_config(&header)
        .unwrap_err()
        .contains("explicit underdog_woof"));
    header
        .0
        .insert("general.architecture".into(), Text(ARCHITECTURE.into()));
    header
        .0
        .insert("underdog_woof.config".into(), Text("{}".into()));
    assert!(validate_config(&header)
        .unwrap_err()
        .contains("configuration"));
    header.0.insert(
        "underdog_woof.config".into(),
        Text(contract()["config"].to_string()),
    );
    header
        .0
        .insert("underdog_woof.format".into(), Text("iq1_s".into()));
    assert!(validate_config(&header)
        .unwrap_err()
        .contains("weight format"));
}

fn read_f32(path: &std::path::Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(bytes.len() % 4, 0);
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

fn metrics(actual: &[f32], expected: &[f32]) -> Value {
    assert_eq!(actual.len(), expected.len());
    assert!(actual.iter().chain(expected).all(|x| x.is_finite()));
    let (mut error, mut dot, mut a2, mut b2, mut max) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (&a, &b) in actual.iter().zip(expected) {
        let (a, b) = (f64::from(a), f64::from(b));
        error += (a - b).powi(2);
        dot += a * b;
        a2 += a * a;
        b2 += b * b;
        max = max.max((a - b).abs());
    }
    serde_json::json!({"max_abs":max,"rmse":(error / actual.len() as f64).sqrt(),
        "relative_rmse":(error / b2.max(1e-30)).sqrt(),"cosine":dot / (a2 * b2).sqrt().max(1e-30)})
}

#[test]
#[ignore = "requires the pinned remote GGUF and official per-format fixtures"]
fn official_pipeline_fixture() {
    let root =
        std::path::PathBuf::from(std::env::var("RMI_WOOF_FIXTURE").expect("RMI_WOOF_FIXTURE"));
    let manifest: Value =
        serde_json::from_slice(&std::fs::read(root.join("manifest.json")).unwrap()).unwrap();
    let path = std::env::var("RMI_WOOF_GGUF").unwrap_or_else(|_| {
        manifest["same_weight_gguf"]["path"]
            .as_str()
            .unwrap()
            .into()
    });
    let policy = match std::env::var("RMI_WOOF_COMPUTE")
        .as_deref()
        .unwrap_or("cpu")
    {
        "cpu" => ComputePolicy::Cpu,
        "auto" => ComputePolicy::Auto,
        "vulkan" => ComputePolicy::Vulkan,
        _ => panic!("invalid compute"),
    };
    let _scope = policy.configure_cli().unwrap();
    let source = Arc::from(
        crate::format::ggufrs::open_model_source(
            std::path::Path::new(&path),
            crate::format::ggufrs::ComponentRole::Llm,
        )
        .unwrap(),
    );
    let model = Model::from_source(source, 2, policy).unwrap();
    let weights = Weights::load(model.source.as_ref()).unwrap();
    let mut run = Runner::new(&weights, model.pool.clone(), policy).unwrap();
    let ids: Vec<u32> = manifest["text_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    assert_eq!(text::encode_text(manifest["text"].as_str().unwrap()), ids);
    let force = manifest["forced_latents"]["path"]
        .as_str()
        .map(|path| read_f32(std::path::Path::new(path)));
    assert!(
        force.is_some() || manifest["eos_disabled"] == false,
        "free-generation fixtures must use natural EOS"
    );
    let start = std::time::Instant::now();
    let latents = run
        .generate(
            manifest["voice"].as_str().unwrap(),
            &ids,
            &mut rng::Noise::new(manifest["seed"].as_u64().unwrap() as u32),
            manifest["temperature"].as_f64().unwrap() as f32,
            force.as_deref(),
        )
        .unwrap();
    let speech_seconds = start.elapsed().as_secs_f64();
    let start = std::time::Instant::now();
    let waveform = run.decode(force.as_deref().unwrap_or(&latents)).unwrap();
    let decoder_seconds = start.elapsed().as_secs_f64();
    let mut results = serde_json::Map::new();
    for (name, values) in [("latents", latents), ("waveform", waveform)] {
        let expected = read_f32(&root.join(format!("{name}.f32")));
        let result = metrics(&values, &expected);
        results.insert(name.into(), result);
    }
    let result = serde_json::json!({"format":model.format,"requested":format!("{policy:?}"),
        "gpu_projections":run.gpu_calls,"cpu_projections":run.cpu_calls,"metrics":results,
        "seconds":{"speech":speech_seconds,"decoder":decoder_seconds},"oracle":root});
    println!("woof_oracle={result}");
    if let Ok(path) = std::env::var("RMI_WOOF_RESULT") {
        std::fs::write(path, serde_json::to_vec_pretty(&result).unwrap()).unwrap();
    }
    if policy == ComputePolicy::Vulkan {
        assert!(
            run.gpu_calls > 0,
            "strict device mode must execute on the device"
        );
    }
    // Fixed before the model probes: float reduction error versus native block-quantized activations.
    let (relative, cosine, maximum) = match model.format.as_str() {
        "f32" => (2e-4, 0.999999, 0.002),
        "f16" | "bf16" => (0.002, 0.99999, 0.02),
        _ => (0.06, 0.998, 0.25),
    };
    for (name, metric) in &results {
        let expected = read_f32(&root.join(format!("{name}.f32")));
        let peak = expected
            .iter()
            .fold(0.0f64, |p, v| p.max(f64::from(v.abs())));
        assert!(
            metric["relative_rmse"].as_f64().unwrap() <= relative
                && metric["cosine"].as_f64().unwrap() >= cosine
                && metric["max_abs"].as_f64().unwrap() <= maximum * (1.0 + peak),
            "{name}: {metric}"
        );
    }
}
