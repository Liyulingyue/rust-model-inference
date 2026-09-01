use rust_model_inference::core::tokenizer::BPETokenizer;
use rust_model_inference::format::ggufrs::ComponentRole;
use rust_model_inference::open_model_source;
use std::sync::Arc;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = &args[1];
    let ids: Vec<u32> = args[2..].iter().filter_map(|s| s.parse().ok()).collect();
    let source = open_model_source(std::path::Path::new(path), ComponentRole::Llm).unwrap();
    let tokenizer = BPETokenizer::from_gguf_metadata(|k| source.metadata(k).cloned()).unwrap();
    for id in ids {
        println!(
            "id={} text={:?}",
            id,
            String::from_utf8_lossy(&tokenizer.decode_bytes(&[id], true))
        );
    }
}
