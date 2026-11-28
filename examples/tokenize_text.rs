//! Print the token IDs the repository's tokenizer produces for a text prompt.
//!
//! The mirror image of `lookup_tokens` (ids -> text). Used by the
//! `tools/oracle/*` harnesses to diff our tokenizer against llama.cpp's on the
//! same bytes, which is the discrete contract every model adaptation has to
//! match before any numeric comparison means anything.
//!
//! ```text
//! cargo run --example tokenize_text -- <model.gguf> <prompt-file> [add_special] [parse_special]
//! ```
use rust_model_inference::core::tokenizer::{BPETokenizer, EncodeOptions};
use rust_model_inference::format::ggufrs::ComponentRole;
use rust_model_inference::open_model_source;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        return Err(
            "usage: tokenize_text <model.gguf> <prompt-file> [add_special] [parse_special]".into(),
        );
    }
    let source = open_model_source(std::path::Path::new(&args[1]), ComponentRole::Llm)?;
    let tokenizer = BPETokenizer::from_gguf_metadata(|k| source.metadata(k).cloned())?;
    let text = std::fs::read_to_string(&args[2])?;
    let add_special = args.get(3).map(|v| v != "0").unwrap_or(false);
    let parse_special = args.get(4).map(|v| v != "0").unwrap_or(true);
    let ids = tokenizer.encode(
        &text,
        EncodeOptions {
            add_special,
            parse_special,
        },
    );
    println!(
        "{}",
        serde_json::json!({
            "text_bytes": text.len(),
            "add_special": add_special,
            "parse_special": parse_special,
            "n_tokens": ids.len(),
            "ids": ids,
        })
    );
    Ok(())
}
