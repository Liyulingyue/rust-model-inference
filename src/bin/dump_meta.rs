use rust_model_inference::GGUFLoader;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("Usage: dump_meta <model.gguf>");
    let loader = GGUFLoader::from_file(&path).expect("open");
    for (k, v) in loader.metadata_entries() {
        println!("{k} = {v:?}");
    }
}
