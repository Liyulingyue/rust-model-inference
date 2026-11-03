//! Same-input isolated F32 checks; no quantization, F16 tables or KV cache.
use rust_model_inference::ops;
use serde_json::json;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    assert!(cfg!(feature = "parity-trace"), "build with parity-trace");
    assert!(ops::scalar_mode(), "set RMI_SCALAR=1");
    let bytes = std::fs::read(std::env::args().nth(1).ok_or("missing fixture")?)?;
    if bytes.len() < 8 || (bytes.len() - 8) % 4 != 0 {
        return Err("invalid fixture".into());
    }
    let n = u32::from_le_bytes(bytes[..4].try_into()?) as usize;
    let rows = u32::from_le_bytes(bytes[4..8].try_into()?) as usize;
    if n == 0 || n > 4096 || rows != 5 || bytes.len() != 8 + n * (rows + 4) * 4 {
        return Err("invalid fixture shape".into());
    }
    let values: Vec<f32> = bytes[8..]
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    let (x, rest) = values.split_at(n);
    let (matrix, rest) = rest.split_at(n * rows);
    let (weight, rest) = rest.split_at(n);
    let (bias, other) = rest.split_at(n);
    let mut matmul = vec![0.; rows];
    ops::kernel::f32::scalar::forward_f32_rows(matrix, x, &mut matmul, n, rows, 0, 1);
    let mut norm = vec![0.; n];
    ops::layer_norm(x, weight, bias, 1e-5, &mut norm);
    let mut rms = vec![0.; n];
    ops::rms_norm(x, weight, &mut rms, 1e-5);
    let mut softmax = x.to_vec();
    ops::softmax_inplace(&mut softmax);
    let mut silu = x.to_vec();
    ops::silu_inplace(&mut silu);
    let mut add = x.to_vec();
    ops::vec_add_into(other, &mut add);
    let mut multiply = x.to_vec();
    ops::vec_mul_inplace(other, &mut multiply);
    let bits = |values: &[f32]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    println!(
        "{}",
        json!({"matmul":bits(&matmul),"layer_norm":bits(&norm),"rms_norm":bits(&rms),
        "softmax":bits(&softmax),"silu":bits(&silu),"add":bits(&add),"multiply":bits(&multiply)})
    );
    Ok(())
}
