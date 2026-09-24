//! Generic model graph: a heterogeneous stack of `Layer` impls driven by a
//! shared scratchpad.
//!
//! Depends only on [`crate::core::traits`] (Layer, ModelConfig, ExecContext).

use crate::core::traits::{ExecContext, Layer, ModelConfig};

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
