//! Tensor load / projection helpers shared by the boundary heads.
//!
//! Every boundary head (candidate encoder, content pooler, marginals, pair
//! scorer, proposer, record head, relation scorer) loads its parameters with
//! the same two routines and projects with the same two routines. They were
//! byte-identical copies in nine to ten files, so a fix to the dim check in
//! [`load_weight`] had to be made ten times. This module is the single copy.

use crate::core::tensor::TensorSource;
use crate::ops::kernel::{QuantizedTensor, Weight};

pub(crate) fn load_vec(
    source: &dyn TensorSource,
    name: &str,
    len: usize,
) -> Result<Vec<f32>, String> {
    crate::core::tensor::load_f32_tensor(source, name, &[len as u64])
        .map_err(|e| format!("{name}: {e}"))
}

pub(crate) fn load_weight<'a>(
    source: &'a dyn TensorSource,
    name: &str,
    n_in: usize,
    n_out: usize,
) -> Result<Weight<'a>, String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("missing tensor {name}"))?;
    if info.dims != [n_in as u64, n_out as u64] {
        return Err(format!(
            "tensor {name} has dims {:?}, expected [{n_in}, {n_out}]",
            info.dims
        ));
    }
    let bytes = source
        .tensor_slice(name)
        .ok_or_else(|| format!("missing tensor data {name}"))?;
    Ok(Weight::from_quantized(QuantizedTensor::from_bytes(
        bytes,
        info.ggml_type,
        n_in,
        n_out,
    )))
}

pub(crate) fn apply_linear_full(
    input: &[f32],
    weight: &Weight<'_>,
    bias: &[f32],
    output: &mut [f32],
) {
    if let Some(rows) = weight.kernel.f32_slice() {
        let n_in = input.len();
        let n_out = output.len();
        debug_assert_eq!(bias.len(), n_out);
        for (out_index, row) in rows.chunks_exact(n_in).take(n_out).enumerate() {
            output[out_index] = crate::ops::dot_f32(row, input, n_in) + bias[out_index];
        }
    } else {
        weight
            .kernel
            .forward(input, output, weight.n_in, weight.n_out);
        for (out, b) in output.iter_mut().zip(bias.iter()) {
            *out += *b;
        }
    }
}

/// Apply a projection to a stack of rows.
///
/// `apply_linear_full` infers both widths from the slice lengths, so it is only
/// valid for a single row — handing it a whole `[rows, in]` block makes it treat
/// the block as one wide vector and read the weight rows at the wrong stride.
///
/// `n_out` is passed explicitly rather than read from `weight.n_out`: for F32
/// tensors `QuantizedTensor::n_rows()` is `usize::from(!data.is_empty())`, i.e.
/// 0 or 1, so `Weight::n_out` carries no shape information at all on the
/// unquantized boundary heads.
pub(crate) fn apply_linear_rows(
    input: &[f32],
    weight: &Weight<'_>,
    bias: &[f32],
    output: &mut [f32],
    n_out: usize,
) {
    let n_in = weight.n_in.max(1);
    debug_assert!(output.len() >= input.len() / n_in * n_out);
    for row in 0..input.len() / n_in {
        apply_linear_full(
            &input[row * n_in..][..n_in],
            weight,
            bias,
            &mut output[row * n_out..][..n_out],
        );
    }
}
