//! Quantized linear layer: dequantize-then-matmul for Q4_K weights.
//!
//! This is an ops-layer type because `forward_dequant` directly calls
//! `ops::quant::dequantize_q4_k_weight`.

use crate::core::tensor::TensorSource;
use crate::core::traits::{ExecContext, Layer};
use crate::ops::quant::dequantize_q4_k_weight;

pub struct QuantizedLinear<'a> {
    weight: &'a [u8],
    #[allow(dead_code)]
    bias: Option<&'a [u8]>,
    in_features: usize,
    out_features: usize,
    layer_name: &'a str,
}

impl<'a> QuantizedLinear<'a> {
    pub fn from_weight_slice(
        weight: &'a [u8],
        bias: Option<&'a [u8]>,
        in_features: usize,
        out_features: usize,
        name: &'a str,
    ) -> Self {
        Self {
            weight,
            bias,
            in_features,
            out_features,
            layer_name: name,
        }
    }

    pub fn from_source<S: TensorSource + ?Sized>(
        source: &'a S,
        weight_name: &str,
        bias_name: Option<&str>,
        in_features: usize,
        out_features: usize,
        name: &'a str,
    ) -> Option<Self> {
        let weight = source.tensor_slice(weight_name)?;
        let bias = bias_name.and_then(|n| source.tensor_slice(n));
        Some(Self {
            weight,
            bias,
            in_features,
            out_features,
            layer_name: name,
        })
    }

    pub fn weight_ptr(&self) -> usize {
        self.weight.as_ptr() as usize
    }

    pub fn weight_len(&self) -> usize {
        self.weight.len()
    }

    pub fn forward_dequant(&self, input: &[f32], output: &mut [f32], scratch: &mut [f32]) {
        let n_elements = self.out_features * self.in_features;
        let dequant_len = n_elements.min(scratch.len());
        dequantize_q4_k_weight(
            self.weight,
            self.out_features,
            self.in_features,
            &mut scratch[..dequant_len],
        );

        let dequant = &scratch[..dequant_len];

        for i in 0..self.out_features {
            let row_offset = i * self.in_features;
            let mut sum = 0.0f32;
            for j in 0..self.in_features {
                sum += dequant[row_offset + j] * input[j];
            }
            output[i] = sum;
        }
    }
}

impl<'a> Layer for QuantizedLinear<'a> {
    fn forward(&self, input: &[f32], output: &mut [f32], ctx: &mut ExecContext) {
        self.forward_dequant(input, output, ctx.scratch);
    }

    fn input_dim(&self) -> usize {
        self.in_features
    }

    fn output_dim(&self) -> usize {
        self.out_features
    }

    fn name(&self) -> &str {
        self.layer_name
    }
}
