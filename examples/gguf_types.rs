//! Print a GGUF's metadata and the type histogram of its tensors.
//!
//! Used to find out what a quantised export actually contains before adapting a
//! model to it: whether the Q4_K export really is Q4_K, and which tensors were
//! left in a wider format.
//!
//! ```text
//! cargo run --example gguf_types -- models/YuE2-gguf/yue2.gguf
//! ```

use rust_model_inference::core::loader::GGUFLoader;
use std::collections::BTreeMap;

fn main() -> std::process::ExitCode {
    let Some(path) = std::env::args().nth(1) else {
        eprintln!("usage: gguf_types <model.gguf> [key_prefix ...]");
        return std::process::ExitCode::FAILURE;
    };
    let filters: Vec<String> = std::env::args().skip(2).collect();
    let loader = match GGUFLoader::from_file(&path) {
        Ok(loader) => loader,
        Err(error) => {
            eprintln!("{path}: {error}");
            return std::process::ExitCode::FAILURE;
        }
    };

    if filters.is_empty() {
        println!("== metadata ==");
        for (key, value) in loader.metadata_entries() {
            // `MetaValue` covers scalars, strings and arrays and has no
            // `Display`, so `Debug` is the one rendering that always applies.
            let text = format!("{value:?}");
            let text = if text.len() > 100 {
                &text[..100]
            } else {
                &text
            };
            println!("  {key} = {text}");
        }
    }

    println!("\n== tensor types ==");
    let mut histogram: BTreeMap<String, (usize, u64)> = BTreeMap::new();
    let mut matched = 0usize;
    for info in loader.tensors() {
        if !filters.is_empty() && !filters.iter().any(|f| info.name.contains(f.as_str())) {
            continue;
        }
        matched += 1;
        let entry = histogram
            .entry(format!("{:?}", info.ggml_type))
            .or_insert((0, 0));
        entry.0 += 1;
        entry.1 += info.checked_n_elements().unwrap_or(0);
    }
    if filters.is_empty() {
        println!("  (no filter: all {} tensors)", loader.tensors().len());
    }
    println!("  {matched} tensors matched");
    for (ty, (count, elements)) in &histogram {
        println!("  {ty:>10}  {count:6} tensors  {elements:>14} elements");
    }
    std::process::ExitCode::SUCCESS
}
