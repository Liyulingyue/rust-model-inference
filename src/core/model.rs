//! Generic model-graph types: [`QuantizedLinear`] (a quantized matmul Layer)
//! and [`ModelGraph`] (a heterogeneous stack of `Layer` impls driven by a
//! shared scratchpad).
//!
//! Depends on [`crate::core::tensor`] (TensorSource) and
//! [`crate::traits`] (Layer, ModelConfig, ExecContext).

use crate::core::tensor::TensorSource;
use crate::core::traits::{ExecContext, Layer, ModelConfig};
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

pub struct ModelGraph<'a> {
    pub config: ModelConfig,
    pub layers: Vec<Box<dyn Layer + 'a>>,
}

impl<'a> ModelGraph<'a> {
    pub fn new(config: ModelConfig) -> Self {
        Self {
            config,
            layers: Vec::new(),
        }
    }

    pub fn add_layer<L: Layer + 'a>(&mut self, layer: L) {
        self.layers.push(Box::new(layer));
    }

    pub fn forward_all(
        &self,
        input: &[f32],
        output: &mut [f32],
        scratch: &mut [f32],
        ctx: &mut ExecContext,
    ) {
        if self.layers.is_empty() {
            return;
        }

        let dim = self.layers[0].output_dim().max(self.layers[0].input_dim());
        let (buf_a, buf_b) = scratch.split_at_mut(dim);

        buf_a[..input.len()].copy_from_slice(input);

        for (i, layer) in self.layers.iter().enumerate() {
            ctx.layer_idx = i as u32;
            if i % 2 == 0 {
                layer.forward(buf_a, buf_b, ctx);
            } else {
                layer.forward(buf_b, buf_a, ctx);
            }
        }

        let last_idx = self.layers.len() - 1;
        let src = if last_idx % 2 == 0 { buf_b } else { buf_a };
        let out_len = output.len().min(src.len());
        output[..out_len].copy_from_slice(&src[..out_len]);
    }
}
