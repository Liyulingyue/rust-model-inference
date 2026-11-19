use super::{
    acoustic::{norm, qkv_parts, rope},
    add, contract, trace, Runner,
};
use crate::ops::nn::{attention_scalar, AttentionSpec};
use serde_json::Value;
use std::collections::BTreeMap;

type States = BTreeMap<String, Vec<f32>>;

fn elu(values: &mut [f32]) {
    for v in values {
        if *v < 0.0 {
            *v = v.exp_m1();
        }
    }
}

impl Runner<'_, '_> {
    fn decoder_transformer(&mut self, mut x: Vec<f32>) -> Result<Vec<f32>, String> {
        let rows = x.len() / 512;
        for layer in 0..2 {
            let prefix = format!("decoder.tf.{layer}");
            let y = norm(
                &x,
                512,
                Some(self.weights.values(&format!("{prefix}.ln1.weight"))),
                Some(self.weights.values(&format!("{prefix}.ln1.bias"))),
                1e-5,
            )?;
            let qkv = self.linear(&format!("{prefix}.qkv"), &y)?;
            let (mut q, mut k, v) = qkv_parts(&qkv, 512);
            let inv = self.weights.values("decoder.inv_freq");
            rope(&mut q, 0, 8, inv);
            rope(&mut k, 0, 8, inv);
            let mut a = Vec::with_capacity(x.len());
            // The source decoder attends to at most 250 preceding/current frames.
            for token in 0..rows {
                let start = (token + 1).saturating_sub(250);
                a.extend(attention_scalar(
                    &q[token * 512..(token + 1) * 512],
                    &k[start * 512..(token + 1) * 512],
                    &v[start * 512..(token + 1) * 512],
                    AttentionSpec {
                        query_tokens: 1,
                        key_tokens: token + 1 - start,
                        query_heads: 8,
                        key_value_heads: 8,
                        head_dim: 64,
                        causal: false,
                        scale: 0.125,
                    },
                )?);
            }
            add(&mut x, &self.linear(&format!("{prefix}.o"), &a)?);
            let y = norm(
                &x,
                512,
                Some(self.weights.values(&format!("{prefix}.ln2.weight"))),
                Some(self.weights.values(&format!("{prefix}.ln2.bias"))),
                1e-5,
            )?;
            let mut y = self.linear(&format!("{prefix}.fc1"), &y)?;
            crate::ops::activation::gelu_erf_inplace(&mut y);
            add(&mut x, &self.linear(&format!("{prefix}.fc2"), &y)?);
            trace(&format!("decoder.transformer.{layer}"), &[rows, 512], &x);
        }
        trace("decoder.transformer", &[rows, 512], &x);
        Ok(x)
    }

    fn conv(
        &mut self,
        name: &str,
        x: &[f32],
        spec: &Value,
        states: &mut States,
        residual: Option<&[f32]>,
    ) -> Result<Vec<f32>, String> {
        let cin = spec["in_channels"].as_u64().unwrap() as usize;
        let cout = spec["out_channels"].as_u64().unwrap() as usize;
        let kernel = spec["kernel_size"].as_u64().unwrap() as usize;
        let rows = x.len() / cin;
        let mut padded = states
            .get(name)
            .cloned()
            .unwrap_or_else(|| vec![0.0; (kernel - 1) * cin]);
        padded.extend_from_slice(x);
        let mut output = self.linear(&format!("{name}.tap.0"), &padded[..rows * cin])?;
        let bias = self.weights.values(&format!("{name}.bias"));
        if let Some(residual) = residual {
            for (i, y) in output.iter_mut().enumerate() {
                *y += residual[i] + bias[i % cout];
            }
        } else {
            for row in output.chunks_exact_mut(cout) {
                add(row, bias);
            }
        }
        for tap in 1..kernel {
            let values = self.linear(
                &format!("{name}.tap.{tap}"),
                &padded[tap * cin..(tap + rows) * cin],
            )?;
            add(&mut output, &values);
        }
        states.insert(name.to_string(), padded[rows * cin..].to_vec());
        Ok(output)
    }

    fn conv_transpose(
        &mut self,
        name: &str,
        x: &[f32],
        spec: &Value,
        states: &mut States,
    ) -> Result<Vec<f32>, String> {
        let cin = spec["in_channels"].as_u64().unwrap() as usize;
        let cout = spec["out_channels"].as_u64().unwrap() as usize;
        let stride = spec["stride"].as_u64().unwrap() as usize;
        let rows = x.len() / cin;
        let sc = stride * cout;
        let z = self.matmul(name, x)?;
        let carry = states.get(name).cloned().unwrap_or_else(|| vec![0.0; sc]);
        let bias = self.weights.values(&format!("{name}.bias"));
        let mut output = vec![0.0; rows * sc];
        for row in 0..rows {
            for c in 0..sc {
                let previous = if row == 0 {
                    carry[c]
                } else {
                    z[(row - 1) * 2 * sc + sc + c]
                };
                output[row * sc + c] = (z[row * 2 * sc + c] + bias[c % cout]) + previous;
            }
        }
        states.insert(name.to_string(), z[(rows - 1) * 2 * sc + sc..].to_vec());
        Ok(output)
    }

    fn seanet(
        &mut self,
        mut x: Vec<f32>,
        start: usize,
        end: usize,
        states: &mut States,
    ) -> Result<Vec<f32>, String> {
        for i in start..end {
            let spec = &contract()["seanet"][i];
            let name = format!("decoder.seanet.{i}");
            match spec["kind"].as_str().unwrap() {
                "elu" => elu(&mut x),
                "conv" => x = self.conv(&name, &x, spec, states, None)?,
                "convt" => x = self.conv_transpose(&name, &x, spec, states)?,
                "res" => {
                    let mut y = x.clone();
                    elu(&mut y);
                    y = self.conv(&format!("{name}.0"), &y, &spec["layers"][0], states, None)?;
                    elu(&mut y);
                    x = self.conv(
                        &format!("{name}.1"),
                        &y,
                        &spec["layers"][1],
                        states,
                        Some(&x),
                    )?;
                }
                _ => unreachable!("pinned SEANet operation"),
            }
            trace(&name, &[x.len()], &x);
        }
        Ok(x)
    }

    pub(super) fn decode(&mut self, latents: &[f32]) -> Result<Vec<f32>, String> {
        if latents.is_empty() || latents.len() % 32 != 0 || latents.iter().any(|v| !v.is_finite()) {
            return Err(
                "Underdog Woof decoder requires finite, nonempty 32-dimensional frames".into(),
            );
        }
        let mean = self.weights.values("latent.mean");
        let std = self.weights.values("latent.std");
        let latents: Vec<f32> = latents
            .iter()
            .enumerate()
            .map(|(i, z)| z * std[i % 32] + mean[i % 32])
            .collect();
        let frames = latents.len() / 32;
        trace("denormalized_latents", &[frames, 32], &latents);
        let x = self.linear("decoder.inp", &latents)?;
        trace("decoder.input", &[frames, 512], &x);
        let x = self.decoder_transformer(x)?;
        let mut states = States::new();
        let x = self.seanet(x, 0, 6, &mut states)?;
        let mut output = Vec::with_capacity(frames * 960);
        // Retain convolution histories/carries across the source runtime's eight-frame chunks.
        for chunk in x.chunks(8 * 48 * 192) {
            output.extend(self.seanet(chunk.to_vec(), 6, 15, &mut states)?);
        }
        if output.len() != frames * 960 {
            return Err("Underdog Woof decoder sample count mismatch".into());
        }
        trace("waveform", &[output.len()], &output);
        Ok(output)
    }
}
