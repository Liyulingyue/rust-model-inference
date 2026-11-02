//! System One JSON I/O for the shared CLI; model dispatch stays in models.
use std::io::{Read, Write};

pub fn run_cli(options: &crate::app::CliOptions, threads: usize) -> Result<(), String> {
    let path = options
        .system_one_request
        .as_deref()
        .ok_or("missing System One request")?;
    let bytes = if path == std::path::Path::new("-") {
        let mut buf = Vec::new();
        std::io::stdin()
            .read_to_end(&mut buf)
            .map_err(|e| e.to_string())?;
        buf
    } else {
        std::fs::read(path)
            .map_err(|e| format!("Read System One request {}: {e}", path.display()))?
    };
    let request = serde_json::from_slice(&bytes)
        .map_err(|e| format!("Invalid System One request JSON: {e}"))?;
    let answer = crate::models::vela3::system_one(
        &options.model,
        options.base_model.as_deref(),
        &request,
        threads,
    )?;
    let mut out = std::io::stdout().lock();
    serde_json::to_writer(&mut out, &answer).map_err(|e| e.to_string())?;
    writeln!(out).map_err(|e| e.to_string())
}
