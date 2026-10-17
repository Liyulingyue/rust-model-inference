//! Print a GGUF's tensor names, shapes and types.
//!
//! ```text
//! cargo run --example gguf_tensor_list -- models/x/y.gguf [name_filter]
//! ```

use rust_model_inference::core::loader::GGUFLoader;
use std::process::ExitCode;

fn main() -> ExitCode {
    let Some(path) = std::env::args().nth(1) else {
        eprintln!("usage: gguf_tensor_list <model.gguf> [filter ...]");
        return ExitCode::FAILURE;
    };
    let filters: Vec<String> = std::env::args().skip(2).collect();
    let Ok(loader) = GGUFLoader::from_file(&path) else {
        eprintln!("{path}: load failed");
        return ExitCode::FAILURE;
    };
    let mut shown = 0usize;
    for info in loader.tensors() {
        if !filters.is_empty() && !filters.iter().any(|f| info.name.contains(f.as_str())) {
            continue;
        }
        shown += 1;
        let shape: Vec<String> = info.dims.iter().map(|d| d.to_string()).collect();
        println!(
            "{:<64} {:<8} [{:>20}]  n_elem={}",
            info.name,
            format!("{:?}", info.ggml_type),
            shape.join(", "),
            info.checked_n_elements().unwrap_or(0),
        );
    }
    println!("-- {shown} tensors shown");
    ExitCode::SUCCESS
}
