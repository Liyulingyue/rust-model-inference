//! Round-trip check for the converter's k-quant payloads.
//!
//! Reads a raw F32 file and the block payload the Python encoder wrote for
//! it, decodes the payload with the runtime's own decoder and prints the
//! error against the source. The two sides are written independently (the
//! encoder in `tools/converter/utils/kquants.py`, the decoder in
//! `src/ops/quant/`), so a layout mistake shows up here as a huge error.
//!
//! Usage: `kquant_roundtrip <q4_k|q5_k|q6_k> <values.f32> <payload.bin>`

use rust_model_inference::ops::quant::{
    dequantize_row_q4_k, dequantize_row_q5_k, dequantize_row_q6_k,
};

fn main() {
    let Some(format) = std::env::args().nth(1) else {
        println!("usage: kquant_roundtrip <q4_k|q5_k|q6_k> <values.f32> <payload.bin>");
        return;
    };
    let Some(values_path) = std::env::args().nth(2) else {
        println!("usage: kquant_roundtrip <q4_k|q5_k|q6_k> <values.f32> <payload.bin>");
        return;
    };
    let Some(payload_path) = std::env::args().nth(3) else {
        println!("usage: kquant_roundtrip <q4_k|q5_k|q6_k> <values.f32> <payload.bin>");
        return;
    };
    let bytes = std::fs::read(&values_path).expect("values");
    let payload = std::fs::read(&payload_path).expect("payload");
    let values: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect();
    let mut decoded = vec![0.0f32; values.len()];
    match format.as_str() {
        "q4_k" => dequantize_row_q4_k(&payload, &mut decoded),
        "q5_k" => dequantize_row_q5_k(&payload, &mut decoded),
        "q6_k" => dequantize_row_q6_k(&payload, &mut decoded),
        other => {
            println!("unknown format {other}");
            return;
        }
    }
    let mut worst = 0.0f32;
    let mut rms = 0.0f64;
    let mut rms_src = 0.0f64;
    for (got, want) in decoded.iter().zip(&values) {
        let error = (got - want).abs();
        worst = worst.max(error);
        rms += (error as f64) * (error as f64);
        rms_src += (*want as f64) * (*want as f64);
    }
    let n = values.len() as f64;
    println!(
        "{format}: values={} worst_abs={:.3e} rms_rel={:.3e}",
        values.len(),
        worst,
        (rms / n).sqrt() / (rms_src / n).sqrt().max(1e-12)
    );
}
