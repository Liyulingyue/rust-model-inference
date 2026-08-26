use image::ImageReader;
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::Command;

const TRACE_FILTER: &str = "z_image.prompt_ids,z_image.initial_noise,z_image.sigmas,z_image.timesteps,z_image.text_layer_35,z_image.dit.prelude.text,z_image.dit.prelude.image,z_image.dit.context_refiner.0,z_image.dit.context_refiner.1,z_image.dit.noise_refiner.0,z_image.dit.noise_refiner.1,z_image.dit.layer.0,z_image.dit.layer.29,z_image.dit.velocity,z_image.final_latent,z_image.vae.mapped_latent,z_image.vae.conv_in,z_image.vae.mid.block_1,z_image.vae.mid.attention,z_image.vae.mid,z_image.vae.up.3,z_image.vae.up.2,z_image.vae.up.1,z_image.vae.up.0,z_image.vae.rgb_channels";

struct Fixture {
    text_layer_35: Vec<f32>,
    final_latent: Vec<f32>,
    rgb: Vec<u8>,
}

fn required(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("missing {name}"))
}

fn run(command: &mut Command) {
    let output = command.output().unwrap();
    assert!(
        output.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

fn records(path: &Path) -> Vec<Value> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

fn named<'a>(records: &'a [Value], name: &str) -> Vec<&'a Value> {
    let selected = records
        .iter()
        .filter(|record| record["name"] == name)
        .collect::<Vec<_>>();
    assert!(!selected.is_empty(), "missing parity checkpoint {name}");
    selected
}

fn shape(record: &Value) -> Vec<usize> {
    record["shape"]
        .as_array()
        .expect("checkpoint must contain a shape")
        .iter()
        .map(|value| value.as_u64().unwrap() as usize)
        .collect()
}

fn token_ids(record: &Value) -> Vec<u32> {
    let values = record["token_ids"]
        .as_array()
        .expect("checkpoint must contain token_ids")
        .iter()
        .map(|value| u32::try_from(value.as_u64().unwrap()).unwrap())
        .collect::<Vec<_>>();
    validate_manifest(record, "token IDs", values.len());
    values
}

fn sidecar(trace: &Path, record: &Value) -> PathBuf {
    if let Some(path) = record["binary_path"].as_str() {
        return PathBuf::from(path);
    }
    trace.parent().unwrap().join(
        record["path"]
            .as_str()
            .expect("checkpoint must name a sidecar"),
    )
}

fn read_i32(path: &Path) -> Vec<u32> {
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(bytes.len() % 4, 0, "{}", path.display());
    bytes
        .chunks_exact(4)
        .map(|bytes| {
            u32::try_from(i32::from_le_bytes(bytes.try_into().unwrap()))
                .expect("oracle token IDs must be nonnegative")
        })
        .collect()
}

fn manifest_len(record: &Value, name: &str) -> usize {
    usize::try_from(
        record["len"]
            .as_u64()
            .unwrap_or_else(|| panic!("{name} checkpoint must contain len")),
    )
    .unwrap_or_else(|_| panic!("{name} checkpoint len does not fit usize"))
}

fn validate_manifest(record: &Value, name: &str, actual_len: usize) {
    let shape = shape(record);
    let shape_len = shape
        .iter()
        .try_fold(1usize, |length, dimension| length.checked_mul(*dimension))
        .unwrap_or_else(|| panic!("{name} checkpoint shape overflows"));
    let declared_len = manifest_len(record, name);
    assert_eq!(shape_len, declared_len, "{name} manifest shape product");
    assert_eq!(actual_len, declared_len, "{name} sidecar length");
}

fn validated_f32_sidecar(trace: &Path, record: &Value, name: &str) -> Vec<f32> {
    let values = read_f32(&sidecar(trace, record));
    validate_manifest(record, name, values.len());
    values
}

fn validated_i32_sidecar(trace: &Path, record: &Value, name: &str) -> Vec<u32> {
    let values = read_i32(&sidecar(trace, record));
    validate_manifest(record, name, values.len());
    values
}

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(bytes.len() % 4, 0, "{}", path.display());
    bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect()
}

fn exact_f32_checkpoint(name: &str, rust_trace: &Path, oracle_trace: &Path) {
    let rust_records = records(rust_trace);
    let oracle_records = records(oracle_trace);
    let rust = named(&rust_records, name);
    let oracle = named(&oracle_records, name);
    assert_eq!(rust.len(), oracle.len(), "{name} occurrence count");
    for (occurrence, (rust, oracle)) in rust.iter().zip(oracle).enumerate() {
        assert_eq!(shape(rust), shape(oracle), "{name} occurrence {occurrence}");
        let rust = validated_f32_sidecar(rust_trace, rust, name);
        let oracle = validated_f32_sidecar(oracle_trace, oracle, name);
        assert_eq!(rust.len(), oracle.len(), "{name} occurrence {occurrence}");
        assert_eq!(
            rust.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
            oracle
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "{name} occurrence {occurrence}",
        );
    }
}

fn compare_f32_checkpoint(name: &str, rust_trace: &Path, oracle_trace: &Path, tolerance: f32) {
    let rust_records = records(rust_trace);
    let oracle_records = records(oracle_trace);
    let rust = named(&rust_records, name);
    let oracle = named(&oracle_records, name);
    assert_eq!(rust.len(), oracle.len(), "{name} occurrence count");
    for (occurrence, (rust_record, oracle_record)) in rust.iter().zip(oracle).enumerate() {
        let rust_shape = shape(rust_record);
        let oracle_shape = shape(oracle_record);
        assert_eq!(rust_shape, oracle_shape, "{name} occurrence {occurrence}");
        let rust = validated_f32_sidecar(rust_trace, rust_record, name);
        let oracle = validated_f32_sidecar(oracle_trace, oracle_record, name);
        assert_eq!(rust.len(), oracle.len(), "{name} occurrence {occurrence}");
        assert!(
            rust.iter().chain(&oracle).all(|value| value.is_finite()),
            "{name} occurrence {occurrence}",
        );
        if let Some((index, (&rust, &oracle))) = rust
            .iter()
            .zip(&oracle)
            .enumerate()
            .find(|(_, (left, right))| (*left - *right).abs() > tolerance)
        {
            panic!(
                "{name} occurrence {occurrence} shape {rust_shape:?} index {index}: Rust={rust} oracle={oracle} abs_error={}",
                (rust - oracle).abs(),
            );
        }
    }
}

fn png_rgb(path: &Path) -> Vec<u8> {
    ImageReader::open(path)
        .unwrap()
        .decode()
        .unwrap()
        .into_rgb8()
        .into_raw()
}

fn oracle_command(
    oracle: &str,
    dit: &str,
    text: &str,
    vae: &str,
    prompt: &str,
    trace: &Path,
    output: &Path,
) -> Command {
    let mut command = Command::new(oracle);
    command
        .env("Z_IMAGE_ORACLE_TRACE", trace)
        .args(["--diffusion-model", dit, "--llm", text, "--vae", vae])
        .arg("--prompt")
        .arg(prompt)
        .args([
            "--steps",
            "8",
            "--width",
            "512",
            "--height",
            "512",
            "--seed",
            "42",
            "--threads",
            "1",
            "--rng",
            "cpu",
            "--cfg-scale",
            "1.0",
            "--sampling-method",
            "euler",
        ])
        .arg("--output")
        .arg(output);
    command
}

fn run_fixture(fixture: usize, prompt: &str) -> Fixture {
    let dit = required("Z_IMAGE_DIT");
    let text = required("Z_IMAGE_TEXT");
    let vae = required("Z_IMAGE_VAE");
    let oracle = required("Z_IMAGE_ORACLE_BIN");
    let trace_root = PathBuf::from(required("Z_IMAGE_ORACLE_TRACE"));
    let directory = trace_root.join(format!("rmi-z-image-{}-{fixture}", std::process::id()));
    let oracle_trace = directory.join("oracle.jsonl");
    let rust_trace = directory.join("rust.jsonl");
    let oracle_png = directory.join("oracle.png");
    let rust_png = directory.join("rust.png");
    let _ = std::fs::remove_dir_all(&directory);
    std::fs::create_dir_all(&directory).unwrap();

    run(&mut oracle_command(
        &oracle,
        &dit,
        &text,
        &vae,
        prompt,
        &oracle_trace,
        &oracle_png,
    ));
    run(Command::new(env!("CARGO_BIN_EXE_rust-model-inference"))
        .env("RMI_PARITY_TRACE", &rust_trace)
        .env("RMI_PARITY_FILTER", TRACE_FILTER)
        .args([
            "--model",
            &dit,
            "--text-encoder",
            &text,
            "--vae",
            &vae,
            "--prompt",
            prompt,
            "--steps",
            "8",
            "--resolution",
            "512",
            "--seed",
            "42",
            "--threads",
            "1",
            "--out",
            rust_png.to_str().unwrap(),
        ]));

    let rust_records = records(&rust_trace);
    let oracle_records = records(&oracle_trace);
    let rust_tokens = named(&rust_records, "z_image.prompt_ids")[0];
    let oracle_tokens = named(&oracle_records, "z_image.prompt_ids")[0];
    assert_eq!(
        token_ids(rust_tokens),
        validated_i32_sidecar(&oracle_trace, oracle_tokens, "z_image.prompt_ids"),
        "z_image.prompt_ids",
    );
    for name in [
        "z_image.initial_noise",
        "z_image.sigmas",
        "z_image.timesteps",
    ] {
        exact_f32_checkpoint(name, &rust_trace, &oracle_trace);
    }
    for name in [
        "z_image.text_layer_35",
        "z_image.dit.prelude.text",
        "z_image.dit.prelude.image",
        "z_image.dit.context_refiner.0",
        "z_image.dit.context_refiner.1",
        "z_image.dit.noise_refiner.0",
        "z_image.dit.noise_refiner.1",
        "z_image.dit.layer.0",
        "z_image.dit.layer.29",
        "z_image.dit.velocity",
        "z_image.vae.mapped_latent",
        "z_image.vae.conv_in",
        "z_image.vae.mid.block_1",
        "z_image.vae.mid.attention",
        "z_image.vae.mid",
        "z_image.vae.up.3",
        "z_image.vae.up.2",
        "z_image.vae.up.1",
        "z_image.vae.up.0",
        "z_image.vae.rgb_channels",
    ] {
        compare_f32_checkpoint(name, &rust_trace, &oracle_trace, 1e-4);
    }
    compare_f32_checkpoint("z_image.final_latent", &rust_trace, &oracle_trace, 1e-3);

    let rust_rgb = png_rgb(&rust_png);
    let oracle_rgb = png_rgb(&oracle_png);
    assert_eq!(rust_rgb.len(), oracle_rgb.len(), "RGB channel count");
    let max_delta = rust_rgb
        .iter()
        .zip(&oracle_rgb)
        .map(|(&rust, &oracle)| rust.abs_diff(oracle))
        .max()
        .unwrap();
    let exact = rust_rgb
        .iter()
        .zip(&oracle_rgb)
        .filter(|(rust, oracle)| rust == oracle)
        .count();
    let exact_rate = exact as f64 / rust_rgb.len() as f64;
    assert!(max_delta <= 1, "RGB max channel delta {max_delta}");
    assert!(
        exact_rate >= 0.999,
        "RGB exact-byte rate {exact_rate:.6} is below 0.999",
    );

    let text_layer_35 = read_f32(&sidecar(
        &rust_trace,
        named(&rust_records, "z_image.text_layer_35")[0],
    ));
    let final_latent = read_f32(&sidecar(
        &rust_trace,
        named(&rust_records, "z_image.final_latent")[0],
    ));
    let _ = std::fs::remove_dir_all(&directory);
    Fixture {
        text_layer_35,
        final_latent,
        rgb: rust_rgb,
    }
}

#[test]
#[ignore = "requires Z_IMAGE_DIT, Z_IMAGE_TEXT, Z_IMAGE_VAE, Z_IMAGE_ORACLE_BIN, and Z_IMAGE_ORACLE_TRACE"]
fn z_image_matches_pinned_oracle_and_changes_with_prompt() {
    let first = run_fixture(0, "A red fox sleeping beneath a pine tree");
    let second = run_fixture(1, "A blue ceramic fox beneath a pine tree");
    assert_ne!(first.text_layer_35, second.text_layer_35);
    assert_ne!(first.final_latent, second.final_latent);
    assert_ne!(first.rgb, second.rgb);
}

#[test]
fn oracle_command_selects_cpu_mt19937_rng() {
    let command = oracle_command(
        "oracle",
        "dit.gguf",
        "text.gguf",
        "vae.gguf",
        "fox",
        Path::new("oracle.jsonl"),
        Path::new("oracle.png"),
    );
    let args = command
        .get_args()
        .map(|argument| argument.to_string_lossy().into_owned())
        .collect::<Vec<_>>();
    assert!(args.windows(2).any(|pair| pair == ["--rng", "cpu"]));
}

fn malformed_trace_root(tag: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "rmi-z-image-malformed-{tag}-{}-{}",
        std::process::id(),
        line!(),
    ));
    let _ = std::fs::remove_dir_all(&path);
    std::fs::create_dir_all(&path).unwrap();
    path
}

fn write_f32_record(
    trace: &Path,
    sidecar_path: &Path,
    shape: &[usize],
    len: usize,
    values: &[f32],
) {
    let bytes = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    std::fs::write(sidecar_path, bytes).unwrap();
    let record = serde_json::json!({
        "name": "malformed",
        "shape": shape,
        "len": len,
        "occurrence": 0,
        "binary_path": sidecar_path,
    });
    std::fs::write(trace, serde_json::to_vec(&record).unwrap()).unwrap();
}

#[test]
fn token_reader_rejects_trailing_partial_word() {
    let root = malformed_trace_root("token-tail");
    let sidecar_path = root.join("tokens.i32");
    std::fs::write(&sidecar_path, [1, 0, 0, 0, 2]).unwrap();

    let rejected = std::panic::catch_unwind(|| read_i32(&sidecar_path)).is_err();

    std::fs::remove_dir_all(root).unwrap();
    assert!(rejected, "token reader accepted a trailing partial word");
}

#[test]
fn f32_reader_rejects_trailing_partial_word() {
    let root = malformed_trace_root("f32-tail");
    let sidecar_path = root.join("values.f32");
    std::fs::write(&sidecar_path, [0, 0, 128, 63, 2]).unwrap();

    let rejected = std::panic::catch_unwind(|| read_f32(&sidecar_path)).is_err();

    std::fs::remove_dir_all(root).unwrap();
    assert!(rejected, "F32 reader accepted a trailing partial word");
}

#[test]
fn exact_checkpoint_rejects_sidecar_shorter_than_manifest_len() {
    let root = malformed_trace_root("short-sidecar");
    let rust_trace = root.join("rust.jsonl");
    let oracle_trace = root.join("oracle.jsonl");
    let rust_sidecar = root.join("rust.f32");
    let oracle_sidecar = root.join("oracle.f32");
    write_f32_record(&rust_trace, &rust_sidecar, &[2], 2, &[1.0]);
    write_f32_record(&oracle_trace, &oracle_sidecar, &[2], 2, &[1.0, 2.0]);
    let rust_rejected =
        std::panic::catch_unwind(|| exact_f32_checkpoint("malformed", &rust_trace, &oracle_trace))
            .is_err();
    write_f32_record(&rust_trace, &rust_sidecar, &[2], 2, &[1.0, 2.0]);
    write_f32_record(&oracle_trace, &oracle_sidecar, &[2], 2, &[1.0]);
    let oracle_rejected =
        std::panic::catch_unwind(|| exact_f32_checkpoint("malformed", &rust_trace, &oracle_trace))
            .is_err();

    std::fs::remove_dir_all(root).unwrap();
    assert!(
        rust_rejected,
        "exact comparator accepted a truncated Rust sidecar"
    );
    assert!(
        oracle_rejected,
        "exact comparator accepted a truncated oracle sidecar"
    );
}

#[test]
fn exact_checkpoint_rejects_shape_product_mismatch() {
    let root = malformed_trace_root("shape-product");
    let rust_trace = root.join("rust.jsonl");
    let oracle_trace = root.join("oracle.jsonl");
    let rust_sidecar = root.join("rust.f32");
    let oracle_sidecar = root.join("oracle.f32");
    write_f32_record(&rust_trace, &rust_sidecar, &[2], 1, &[1.0]);
    write_f32_record(&oracle_trace, &oracle_sidecar, &[2], 2, &[1.0, 2.0]);
    let rust_rejected =
        std::panic::catch_unwind(|| exact_f32_checkpoint("malformed", &rust_trace, &oracle_trace))
            .is_err();
    write_f32_record(&rust_trace, &rust_sidecar, &[2], 2, &[1.0, 2.0]);
    write_f32_record(&oracle_trace, &oracle_sidecar, &[2], 1, &[1.0]);
    let oracle_rejected =
        std::panic::catch_unwind(|| exact_f32_checkpoint("malformed", &rust_trace, &oracle_trace))
            .is_err();

    std::fs::remove_dir_all(root).unwrap();
    assert!(
        rust_rejected,
        "exact comparator accepted an invalid Rust shape"
    );
    assert!(
        oracle_rejected,
        "exact comparator accepted an invalid oracle shape"
    );
}
