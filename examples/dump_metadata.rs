use rust_model_inference::GGUFLoader;
fn main() {
    let path = std::env::args().nth(1).expect("path");
    let loader = GGUFLoader::from_file(path).unwrap();
    let mut counts: std::collections::BTreeMap<String, usize> = Default::default();
    let mut per_layer_counts: std::collections::BTreeMap<usize, usize> = Default::default();
    for t in loader.tensors() {
        *counts.entry(format!("{:?}", t.ggml_type)).or_insert(0) += 1;
        if let Some(rest) = t.name.strip_prefix("blk.") {
            if let Some(idx) = rest.split('.').next().and_then(|s| s.parse::<usize>().ok()) {
                *per_layer_counts.entry(idx).or_insert(0) += 1;
            }
        }
    }
    println!("TYPES:");
    for (ty, n) in counts.iter() { println!("  {ty}: {n}"); }
    println!("PER-LAYER: {} layers, counts:", per_layer_counts.len());
    for (idx, n) in per_layer_counts.iter() {
        println!("  blk.{idx}: {n} tensors");
    }
}
