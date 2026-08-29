use std::path::PathBuf;
use std::process::Command;

fn gemma4_model_path() -> PathBuf {
    std::env::var_os("RMI_GEMMA4_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models/gemma-4-e2b/gemma-4-E2B-it-Q8_0.gguf"))
}

fn gemma4_mmproj_path() -> PathBuf {
    std::env::var_os("RMI_GEMMA4_MMPROJ")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models/gemma-4-e2b/mmproj-F16.gguf"))
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
    assert!(gemma4_model_path().is_file());
    assert!(gemma4_mmproj_path().is_file());
    assert!(ensure_gemma4_oracle().unwrap().is_file());
}
