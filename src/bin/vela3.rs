//! System One JSON entry point for the six Vela3 model packages.
use std::io::{Read, Write};
use std::path::PathBuf;

fn run() -> Result<(), String> {
    let mut model = None;
    let mut base = None;
    let mut request = None;
    let mut threads = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1);
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--help" || arg == "-h" {
            println!("vela3 --model DIR [--base-model DIR] [--request FILE|-] [--threads N]\n\nRead a System One JSON request from FILE or stdin; write one JSON response to stdout.");
            return Ok(());
        }
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {arg}"))?;
        match arg.as_str() {
            "--model" => model = Some(PathBuf::from(value)),
            "--base-model" => base = Some(PathBuf::from(value)),
            "--request" => request = Some(value),
            "--threads" => {
                threads = value
                    .parse::<usize>()
                    .ok()
                    .filter(|&n| n > 0)
                    .ok_or("threads must be a positive integer")?
            }
            _ => return Err(format!("unknown argument {arg}")),
        }
    }
    let model = model.ok_or("--model DIR is required (see --help)")?;
    let bytes = match request.as_deref() {
        None | Some("-") => {
            let mut buf = Vec::new();
            std::io::stdin()
                .read_to_end(&mut buf)
                .map_err(|e| e.to_string())?;
            buf
        }
        Some(file) => std::fs::read(file).map_err(|e| format!("Read request {file}: {e}"))?,
    };
    let request =
        serde_json::from_slice(&bytes).map_err(|e| format!("Invalid request JSON: {e}"))?;
    let answer = rust_model_inference::models::vela3::system_one(
        &model,
        base.as_deref(),
        &request,
        threads,
    )?;
    let mut out = std::io::stdout().lock();
    serde_json::to_writer(&mut out, &answer).map_err(|e| e.to_string())?;
    writeln!(out).map_err(|e| e.to_string())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("vela3: {error}");
        std::process::exit(1);
    }
}
