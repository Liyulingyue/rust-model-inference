//! Probe: compare the causal-prefix softmax against the padded one.

use rust_model_inference::ops::kernel::f32::ggml;
use rust_model_inference::ops::softmax_ggml_inplace;

#[test]
fn probe_padded_softmax() {
    let s0 = 0.5f32;
    // Old shape: only the causal prefix.
    let mut a = vec![s0];
    softmax_ggml_inplace(&mut a);
    // New shape: padded to 16 columns, the tail masked with -inf.
    let mut b = vec![f32::NEG_INFINITY; 16];
    b[0] = s0;
    softmax_ggml_inplace(&mut b);
    println!("old: {:?}", &a[..1]);
    println!("new: {:?}", &b[..4]);

    // The dot over the padded key count.
    let v: Vec<f32> = (0..16).map(|i| (i as f32) * 0.5 - 1.0).collect();
    let stride = 16;
    let old_dot = ggml::dot_f32_ggml_stride(&v, stride, &a);
    let new_dot = ggml::dot_f32_ggml_stride_padded(&v, stride, &b, 1);
    println!("old_dot = {old_dot:?}  new_dot = {new_dot:?}");
    assert_eq!(old_dot.to_bits(), new_dot.to_bits());
}
