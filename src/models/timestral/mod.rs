//! Native F32 inference for the published Timestral-209M checkpoint.
mod forecast;
#[cfg(feature = "vulkan")]
mod gpu;
mod network;

use crate::core::tensor::{GGMLType, MetaValue, TensorSource};
pub use forecast::{Forecast, ForecastRequest, Series};
use rayon::prelude::*;
use std::collections::HashMap;

const CONFIG_JSON: &str = include_str!("config.json");
const D: usize = 768;
const HEADS: usize = 12;
const HD: usize = D / HEADS;
const F: usize = 2048;
const PS: usize = 32;
const Q: usize = 21;
const MEDIAN: usize = 10;
const LEVELS: [f32; Q] = [
    0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8,
    0.85, 0.9, 0.95, 0.99,
];

pub struct TimestralModel<'a> {
    weights: HashMap<String, &'a [f32]>,
    compute: crate::compute::ComputePolicy,
    #[cfg(feature = "vulkan")]
    gpu: std::sync::Mutex<Option<gpu::Transformer>>,
}

fn tensor_contracts() -> Vec<(String, Vec<u64>)> {
    let mut shapes = vec![
        ("representation.forecast_query".into(), vec![D as u64]),
        ("transformer.out_norm.scale".into(), vec![D as u64]),
    ];
    for (prefix, inp, out, bias) in [
        ("representation.patch_proj", 2 * PS, D, true),
        ("head.projection", D, PS * Q, false),
    ] {
        for (name, n_in, n_out) in [
            ("hidden_layer", inp, 4 * D),
            ("output_layer", 4 * D, out),
            ("residual_layer", inp, out),
        ] {
            shapes.push((
                format!("{prefix}.{name}.weight"),
                vec![n_in as u64, n_out as u64],
            ));
            if bias {
                shapes.push((format!("{prefix}.{name}.bias"), vec![n_out as u64]));
            }
        }
    }
    for i in 0..24 {
        let p = format!("transformer.layers.{i}");
        shapes.extend([
            (format!("{p}.mlp_norm.scale"), vec![D as u64]),
            (
                format!("{p}.mlp.w12.weight"),
                vec![D as u64, (2 * F) as u64],
            ),
            (format!("{p}.mlp.w3.weight"), vec![F as u64, D as u64]),
        ]);
        for j in 0..if i % 4 == 3 { 2 } else { 1 } {
            shapes.push((format!("{p}.attn_norms.{j}.scale"), vec![D as u64]));
            for (name, n_out) in [("wQKV", 3 * D), ("wO", D), ("_gate", D)] {
                shapes.push((
                    format!("{p}.attns.{j}.{name}.weight"),
                    vec![D as u64, n_out as u64],
                ));
            }
        }
    }
    shapes
}

impl<'a> TimestralModel<'a> {
    pub fn from_source(source: &'a dyn TensorSource) -> Result<Self, String> {
        Self::from_source_with_compute(source, crate::compute::ComputePolicy::legacy())
    }

    pub fn from_source_with_compute(
        source: &'a dyn TensorSource,
        compute: crate::compute::ComputePolicy,
    ) -> Result<Self, String> {
        compute.check_build()?;
        let arch = source
            .metadata("general.architecture")
            .and_then(MetaValue::to_string_val);
        if arch != Some("timestral") {
            return Err("Expected timestral architecture".into());
        }
        let raw = source
            .metadata("timestral.config")
            .and_then(MetaValue::to_string_val)
            .ok_or("Missing timestral.config")?;
        let config: serde_json::Value = serde_json::from_str(raw).map_err(|e| e.to_string())?;
        if config != serde_json::from_str::<serde_json::Value>(CONFIG_JSON).unwrap() {
            return Err("Only the published Timestral-209M configuration is supported".into());
        }
        let mut weights = HashMap::new();
        for (name, dims) in tensor_contracts() {
            let info = source
                .tensor_info(&name)
                .ok_or_else(|| format!("Missing tensor {name}"))?;
            if info.dims != dims || info.ggml_type != GGMLType::F32 {
                return Err(format!("Invalid F32 tensor shape/type {name}"));
            }
            let bytes = source
                .tensor_slice(&name)
                .ok_or_else(|| format!("Missing tensor data {name}"))?;
            let values: &[f32] = bytemuck::try_cast_slice(bytes)
                .map_err(|e| format!("Invalid tensor {name}: {e}"))?;
            if values.len() != dims.iter().product::<u64>() as usize
                || !values.iter().all(|v| v.is_finite())
            {
                return Err(format!("Invalid tensor data {name}"));
            }
            weights.insert(name, values);
        }
        Ok(Self {
            weights,
            compute,
            #[cfg(feature = "vulkan")]
            gpu: std::sync::Mutex::new(None),
        })
    }

    fn linear(&self, prefix: &str, input: &[f32], n_in: usize, n_out: usize) -> Vec<f32> {
        let w = self.weights[&format!("{prefix}.weight")];
        let bias = self.weights.get(&format!("{prefix}.bias"));
        let mut output = vec![0.; input.len() / n_in * n_out];
        output
            .par_chunks_mut(n_out)
            .zip(input.par_chunks(n_in))
            .for_each(|(out, inp)| {
                crate::ops::kernel::f32::forward_f32_rows_dispatch(w, inp, out, n_in, n_out, 0, 1);
                if let Some(b) = bias {
                    for (v, b) in out.iter_mut().zip(*b) {
                        *v += b;
                    }
                }
            });
        output
    }

    fn norm(&self, prefix: &str, input: &[f32]) -> Vec<f32> {
        let w = self.weights[&format!("{prefix}.scale")];
        let mut out = vec![0.; input.len()];
        out.par_chunks_mut(D)
            .zip(input.par_chunks(D))
            .for_each(|(out, inp)| {
                crate::ops::rms_norm(inp, w, out, 1e-6);
            });
        out
    }

    fn residual_block(&self, prefix: &str, input: &[f32], n_in: usize, n_out: usize) -> Vec<f32> {
        let mut hidden = self.linear(&format!("{prefix}.hidden_layer"), input, n_in, 4 * D);
        crate::ops::silu_inplace(&mut hidden);
        let mut out = self.linear(&format!("{prefix}.output_layer"), &hidden, 4 * D, n_out);
        let residual = self.linear(&format!("{prefix}.residual_layer"), input, n_in, n_out);
        add(&mut out, &residual);
        out
    }
}

fn add(out: &mut [f32], rhs: &[f32]) {
    for (a, b) in out.iter_mut().zip(rhs) {
        *a += b;
    }
}

fn trace(name: &str, shape: &[usize], values: &[f32]) -> Result<(), String> {
    #[cfg(feature = "parity-trace")]
    if std::env::var_os("RMI_PARITY_TRACE").is_some() {
        crate::parity_trace::checkpoint(&format!("timestral.{name}"), None, shape, values)
            .map_err(|e| e.to_string())?;
    }
    let _ = (name, shape, values);
    Ok(())
}

#[cfg(test)]
mod tests;
