use rust_model_inference::core::tensor::{GGMLType, TensorInfo};
use rust_model_inference::ops::quant::{dequant_q6k_weight, dequant_weight_q4k};
use std::io::Read;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mut data = Vec::new();
    std::fs::File::open(&a[1])
        .unwrap()
        .read_to_end(&mut data)
        .unwrap();
    let n_cols: usize = a[3].parse().unwrap();
    let n_rows: usize = a[4].parse().unwrap();
    let out = if a[2] == "q4_k" {
        let ti = TensorInfo {
            name: "x".into(),
            dims: vec![n_cols as u64, n_rows as u64],
            ggml_type: GGMLType::Q4K,
            offset: 0,
        };
        dequant_weight_q4k(&data, &ti).expect("q4k dequant")
    } else {
        dequant_q6k_weight(&data, n_cols, n_rows)
    };
    for (i, v) in out.iter().enumerate() {
        print!("{:08x} ", v.to_bits());
        if i % 8 == 7 {
            println!();
        }
    }
}
