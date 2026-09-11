use rust_model_inference::GGUFLoader;

fn main() {
    let path = std::env::args().nth(1).expect("path");
    let loader = GGUFLoader::from_file(&path).expect("open");
    for (k, v) in loader.metadata_entries() {
        if k == "gemma4.attention.sliding_window_pattern"
            || k == "gemma4.feed_forward_length"
            || k == "gemma4.block_count"
            || k == "gemma4.embedding_length"
            || k == "gemma4.attention.head_count_kv"
            || k == "gemma4.attention.shared_kv_layers"
        {
            println!("{k} = {v:?}");
        }
    }
}