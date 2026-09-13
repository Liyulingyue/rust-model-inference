use rust_model_inference::models::qwen_drive::planning::PlanningExpert;
use rust_model_inference::models::qwen_drive::scene::PlanningScene;
use rust_model_inference::{
    bf16_to_f32, f32_to_bf16, ComputePool, GGUFLoader, Qwen35DenseKvSnapshot,
};
use serde::Deserialize;
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

const OFFICIAL_PIN: &str = "28091c1532e869bc7aee91fc0aef6b3e6fd0b2e0";

#[derive(Deserialize)]
struct PlannerTrajectories {
    sft: Vec<u32>,
    rl: Vec<u32>,
}

fn bf16(value: f32) -> f32 {
    bf16_to_f32(f32_to_bf16(value))
}

fn scene_cache() -> Vec<Qwen35DenseKvSnapshot> {
    (0..8)
        .map(|index| {
            let base = (0..3 * 4 * 256)
                .map(|offset| {
                    bf16(((offset % 257) as f32 - 128.0) * 0.0078125 + index as f32 * 0.03125)
                })
                .collect::<Vec<_>>();
            let value = (0..3 * 4 * 256)
                .map(|offset| {
                    let base = ((offset % 257) as f32 - 128.0) * 0.0078125 + index as f32 * 0.03125;
                    bf16(-base * 0.5 + index as f32 * 0.015625)
                })
                .collect::<Vec<_>>();
            Qwen35DenseKvSnapshot {
                layer: index * 4 + 3,
                tokens: 3,
                kv_heads: 4,
                head_dim: 256,
                key: base,
                value,
            }
        })
        .collect()
}

fn scene() -> PlanningScene {
    PlanningScene {
        token: "planner-parity".into(),
        views: Vec::new(),
        instruction: "plan".into(),
        history: (0..16)
            .map(|index| {
                [
                    index as f32 * 0.25,
                    index as f32 * -0.125,
                    -3.5 + index as f32 * 0.5,
                ]
            })
            .collect(),
        history_velocity: (0..16)
            .map(|index| [index as f32 * 0.03125, index as f32 * -0.015625])
            .collect(),
        history_acceleration: (0..16)
            .map(|index| {
                [
                    (index % 5) as f32 * 0.0625 - 0.125,
                    (index % 3) as f32 * 0.03125 - 0.03125,
                ]
            })
            .collect(),
        nav_command: 1,
        ego_status: [0.25, -0.5, 1.0, -1.5, 2.0, -2.5, 0.125, -0.0625],
    }
}

fn trace_records(path: &Path) -> Result<Vec<Value>, String> {
    std::fs::read_to_string(path)
        .map_err(|error| format!("failed to read {}: {error}", path.display()))?
        .lines()
        .map(|line| serde_json::from_str(line).map_err(|error| error.to_string()))
        .collect()
}

fn compare_trace_files(actual: &Path, expected: &Path) -> Result<(), String> {
    let actual = trace_records(actual)?;
    let expected = trace_records(expected)?;
    if actual.len() != expected.len() {
        return Err(format!(
            "planner checkpoint count Rust={} Oracle={}",
            actual.len(),
            expected.len()
        ));
    }
    for (record_index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
        for field in [
            "name",
            "layer",
            "step",
            "shape",
            "len",
            "finite",
            "occurrence",
        ] {
            if actual.get(field) != expected.get(field) {
                return Err(format!(
                    "planner checkpoint {record_index} field {field}: Rust={:?} Oracle={:?}",
                    actual.get(field),
                    expected.get(field)
                ));
            }
        }
        let sidecar = |record: &Value| -> Result<Vec<u8>, String> {
            let path = record
                .get("binary_path")
                .and_then(Value::as_str)
                .ok_or_else(|| format!("planner checkpoint has no sidecar: {record}"))?;
            std::fs::read(path).map_err(|error| format!("failed to read {path}: {error}"))
        };
        let actual_bytes = sidecar(actual)?;
        let expected_bytes = sidecar(expected)?;
        if actual_bytes.len() != expected_bytes.len() {
            return Err(format!(
                "planner checkpoint {} byte count Rust={} Oracle={}",
                actual["name"],
                actual_bytes.len(),
                expected_bytes.len()
            ));
        }
        if actual_bytes.len() % 4 != 0 {
            return Err(format!(
                "planner checkpoint {} has a partial F32 word",
                actual["name"]
            ));
        }
        let name = &actual["name"];
        if let Some((word, (actual, expected))) = actual_bytes
            .chunks_exact(4)
            .zip(expected_bytes.chunks_exact(4))
            .enumerate()
            .find(|(_, (actual, expected))| actual != expected)
        {
            let actual = u32::from_le_bytes(actual.try_into().unwrap());
            let expected = u32::from_le_bytes(expected.try_into().unwrap());
            return Err(format!(
                "planner checkpoint {name} word {word}: Rust=0x{actual:08x} Oracle=0x{expected:08x}"
            ));
        }
    }
    Ok(())
}

fn required_path(name: &str) -> PathBuf {
    let path =
        PathBuf::from(std::env::var_os(name).unwrap_or_else(|| panic!("{name} is required")));
    assert!(path.exists(), "{} does not exist", path.display());
    path
}

fn unique_temp_dir(prefix: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!("{prefix}-{}-{nonce}", std::process::id()));
    std::fs::create_dir(&path).unwrap();
    path
}

fn command_output(command: &mut Command, label: &str) -> Output {
    let rendered = format!("{command:?}");
    let output = command
        .output()
        .unwrap_or_else(|error| panic!("failed to run {label} ({rendered}): {error}"));
    assert!(
        output.status.success(),
        "{label} failed ({rendered})\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

fn planner_kind(model: &Path) -> &'static str {
    let name = model.file_name().unwrap().to_string_lossy();
    if name.contains("planner-sft") {
        "sft"
    } else if name.contains("planner-rl") {
        "rl"
    } else {
        panic!(
            "planner GGUF name must identify sft or rl: {}",
            model.display()
        );
    }
}

fn run_oracle(
    source: &Path,
    model_root: &Path,
    planner: &str,
    steps: usize,
    filter: Option<&str>,
    trace: &Path,
) {
    let script =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tools/qwen_drive/qwen_drive_oracle.py");
    let mut command = Command::new("uv");
    command.args([
        "run",
        "--no-project",
        "--python",
        "3.13",
        "--with",
        "torch==2.8.0",
        "--with",
        "torchvision==0.23.0",
        "--with",
        "numpy<3",
        "--with",
        "safetensors==0.8.0",
        "--with",
        "transformers==5.14.1",
        "--with",
        "pillow",
        "python",
    ]);
    command
        .arg(script)
        .arg("planner")
        .arg("--source")
        .arg(source)
        .args(["--expected-commit", OFFICIAL_PIN])
        .arg("--model-root")
        .arg(model_root)
        .args([
            "--planner",
            planner,
            "--steps",
            &steps.to_string(),
            "--threads",
            "12",
            "--trace",
        ])
        .arg(trace);
    if let Some(filter) = filter {
        command.env("RMI_PARITY_FILTER", filter);
    } else {
        command.env_remove("RMI_PARITY_FILTER");
    }
    command_output(&mut command, "official Qwen-Drive planner Oracle");
}

#[cfg(feature = "parity-trace")]
#[test]
#[ignore = "requires RMI_QWEN_DRIVE_PLANNER and about two GiB of weights"]
fn qwen_drive_planner_real_weight_trace() {
    let path = required_path("RMI_QWEN_DRIVE_PLANNER");
    let source_path = required_path("RMI_QWEN_DRIVE_SOURCE");
    let steps = std::env::var("RMI_QWEN_DRIVE_PLANNER_STEPS")
        .ok()
        .map(|value| value.parse::<usize>().expect("valid planner step count"))
        .unwrap_or(1);
    let artifacts = unique_temp_dir("rmi-qwen-drive-planner");
    let oracle_trace = artifacts.join("oracle.jsonl");
    let rust_trace = artifacts.join("rust.jsonl");
    let planner = planner_kind(&path);
    let filter = (steps > 1).then_some(
        "qwen_drive.planner.noise,qwen_drive.planner.euler,qwen_drive.planner.trajectory",
    );
    run_oracle(
        &source_path,
        path.parent().unwrap(),
        planner,
        steps,
        filter,
        &oracle_trace,
    );
    std::env::set_var("RMI_PARITY_TRACE", &rust_trace);
    if let Some(filter) = filter {
        std::env::set_var("RMI_PARITY_FILTER", filter);
    } else {
        std::env::remove_var("RMI_PARITY_FILTER");
    }
    let source = GGUFLoader::from_file(&path).expect("load planner GGUF");
    let expert = PlanningExpert::from_source(&source).expect("load planning expert");
    let output = expert
        .sample(
            &scene_cache(),
            [257, 258, 259],
            &scene(),
            1,
            steps,
            42,
            &ComputePool::new(12),
        )
        .expect("sample trajectory");
    assert_eq!(
        (output.samples, output.points, output.values.len()),
        (1, 50, 150)
    );
    assert!(output.values.iter().all(|value| value.is_finite()));
    if steps == 10 {
        let fixtures: PlannerTrajectories = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/qwen_drive/planner-trajectories.json"
        )))
        .unwrap();
        let expected = if planner == "sft" {
            fixtures.sft
        } else {
            fixtures.rl
        };
        assert_eq!(
            output
                .values
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected,
            "{planner} 10-step trajectory"
        );
    }
    if let Err(error) = compare_trace_files(&rust_trace, &oracle_trace) {
        panic!("{error}\nartifacts retained in {}", artifacts.display());
    }
    std::fs::remove_dir_all(artifacts).unwrap();
}

#[test]
fn planner_trace_comparison_rejects_one_ulp() {
    let directory = std::env::temp_dir().join(format!(
        "rmi-qwen-drive-trace-{}-{}",
        std::process::id(),
        line!()
    ));
    std::fs::create_dir(&directory).unwrap();
    let actual = directory.join("actual.f32");
    let expected = directory.join("expected.f32");
    std::fs::write(&actual, 1.0f32.to_le_bytes()).unwrap();
    std::fs::write(&expected, (1.0f32.to_bits() + 1).to_le_bytes()).unwrap();
    let record = |path: &Path| {
        serde_json::json!({
            "name": "qwen_drive.planner.trajectory",
            "layer": null,
            "step": null,
            "shape": [1],
            "len": 1,
            "finite": true,
            "occurrence": 0,
            "binary_path": path,
        })
        .to_string()
    };
    let actual_trace = directory.join("actual.jsonl");
    let expected_trace = directory.join("expected.jsonl");
    std::fs::write(&actual_trace, record(&actual)).unwrap();
    std::fs::write(&expected_trace, record(&expected)).unwrap();

    let error = compare_trace_files(&actual_trace, &expected_trace).unwrap_err();
    assert!(error.contains("0x3f800000"));
    assert!(error.contains("0x3f800001"));
    std::fs::remove_dir_all(directory).unwrap();
}
