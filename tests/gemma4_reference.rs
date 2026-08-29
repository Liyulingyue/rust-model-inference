use rust_model_inference::{GGMLType, GGUFLoader, MetaValue};
use std::path::{Path, PathBuf};
use std::process::Command;

const GEMMA4_MODEL_NAME: &str = "gemma-4-E2B-it-Q8_0.gguf";
const GEMMA4_MMPROJ_NAME: &str = "mmproj-F16.gguf";

fn gemma4_model_path() -> PathBuf {
    std::env::var_os("RMI_GEMMA4_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models/gemma-4-e2b").join(GEMMA4_MODEL_NAME))
}

fn gemma4_mmproj_path() -> PathBuf {
    std::env::var_os("RMI_GEMMA4_MMPROJ")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models/gemma-4-e2b").join(GEMMA4_MMPROJ_NAME))
}

fn require_gemma4_gguf(
    path: &Path,
    expected_name: &str,
    architecture: &str,
    tensor: &str,
    ggml_type: GGMLType,
) {
    assert_eq!(
        path.file_name().and_then(|name| name.to_str()),
        Some(expected_name)
    );
    let loader = GGUFLoader::from_file(path).unwrap();
    assert!(
        matches!(loader.metadata("general.architecture"), Some(MetaValue::String(value)) if value == architecture)
    );
    assert_eq!(
        loader.tensor_info(tensor).map(|info| info.ggml_type),
        Some(ggml_type),
        "invalid {tensor}"
    );
}

#[test]
fn gemma4_preflight_rejects_arbitrary_files() {
    assert_ne!(
        Path::new("/etc/passwd")
            .file_name()
            .and_then(|name| name.to_str()),
        Some(GEMMA4_MODEL_NAME)
    );
    assert_ne!(
        Path::new("/etc/passwd")
            .file_name()
            .and_then(|name| name.to_str()),
        Some(GEMMA4_MMPROJ_NAME)
    );
}

fn ensure_gemma4_oracle() -> Result<PathBuf, String> {
    if let Some(path) = std::env::var_os("LLAMA_GEMMA4_TRACE_BIN").map(PathBuf::from) {
        if path.is_file() {
            return Ok(path);
        }
    }

    let llama_dir = std::env::var_os("LLAMA_CPP_DIR").ok_or_else(|| {
        "LLAMA_CPP_DIR is required when LLAMA_GEMMA4_TRACE_BIN is not a file".to_owned()
    })?;
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tools/gemma4/build_oracle.sh");
    let output = Command::new("bash")
        .arg(&script)
        .arg(llama_dir)
        .output()
        .map_err(|error| format!("failed to run {}: {error}", script.display()))?;
    if !output.status.success() {
        return Err(format!(
            "Gemma4 oracle build failed\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        ));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let lines: Vec<_> = stdout
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect();
    if lines.len() != 1 {
        return Err(format!(
            "Gemma4 oracle builder must print exactly one path, got {} non-empty lines",
            lines.len()
        ));
    }
    let path = PathBuf::from(lines[0]);
    if !path.is_file() {
        return Err(format!(
            "Gemma4 oracle builder printed a non-file: {}",
            path.display()
        ));
    }
    Ok(path)
}

#[test]
#[ignore = "requires Gemma4 GGUFs and pinned llama.cpp trace binary"]
fn gemma4_matches_pinned_cpu_oracle() {
    require_gemma4_gguf(
        &gemma4_model_path(),
        GEMMA4_MODEL_NAME,
        "gemma4",
        "token_embd.weight",
        GGMLType::Q8_0,
    );
    require_gemma4_gguf(
        &gemma4_mmproj_path(),
        GEMMA4_MMPROJ_NAME,
        "clip",
        "v.patch_embd.weight",
        GGMLType::F16,
    );
    assert!(ensure_gemma4_oracle().unwrap().is_file());
}
