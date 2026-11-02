//! Native primitives shared only by Idol's components.
use crate::core::tensor::{GGMLType, TensorSource};
use rayon::prelude::*;

pub(super) fn tensor(
    s: &dyn TensorSource,
    name: &str,
    shape: &[usize],
) -> Result<Vec<f32>, String> {
    let info = s
        .tensor_info(name)
        .ok_or_else(|| format!("Missing Idol tensor {name}"))?;
    let dims: Vec<_> = if shape.is_empty() {
        vec![1]
    } else {
        shape.iter().rev().map(|&d| d as u64).collect()
    };
    if info.dims != dims {
        return Err(format!(
            "Idol tensor {name}: shape {:?}, expected {dims:?}",
            info.dims
        ));
    }
    let bytes = s
        .tensor_slice(name)
        .ok_or_else(|| format!("Missing Idol tensor data {name}"))?;
    if info.checked_nbytes() != Some(bytes.len() as u64) {
        return Err(format!("Invalid Idol tensor length {name}"));
    }
    let values: Vec<f32> = match info.ggml_type {
        GGMLType::F32 => bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect(),
        GGMLType::F16 => bytes
            .chunks_exact(2)
            .map(|b| half::f16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32())
            .collect(),
        GGMLType::BF16 => bytes
            .chunks_exact(2)
            .map(|b| half::bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32())
            .collect(),
        _ => return Err(format!("Idol tensor {name} must be F32/F16/BF16")),
    };
    finite(&values, name)?;
    Ok(values)
}
pub(super) fn finite(x: &[f32], name: &str) -> Result<(), String> {
    if x.iter().any(|v| !v.is_finite()) {
        return Err(format!("Idol {name} contains NaN or infinity"));
    }
    Ok(())
}
pub(super) fn round_f16(x: &mut [f32]) {
    x.iter_mut()
        .for_each(|v| *v = half::f16::from_f32(*v).to_f32());
}
pub(super) fn round_bf16(x: &mut [f32]) {
    x.iter_mut()
        .for_each(|v| *v = half::bf16::from_f32(*v).to_f32());
}
pub(super) struct Linear {
    pub input: usize,
    pub output: usize,
    weight: Vec<f32>,
    bias: Vec<f32>,
}
impl Linear {
    pub fn load(
        s: &dyn TensorSource,
        p: &str,
        input: usize,
        output: usize,
        bias: bool,
    ) -> Result<Self, String> {
        if input == 0 || output == 0 {
            return Err("Idol linear dimensions must be positive".into());
        }
        Ok(Self {
            input,
            output,
            weight: tensor(s, &format!("{p}.weight"), &[output, input])?,
            bias: if bias {
                tensor(s, &format!("{p}.bias"), &[output])?
            } else {
                vec![0.; output]
            },
        })
    }
    pub fn to_f16(&mut self) -> Result<(), String> {
        round_f16(&mut self.weight);
        round_f16(&mut self.bias);
        finite(&self.weight, "FP16 linear weight")?;
        finite(&self.bias, "FP16 linear bias")
    }
    pub fn to_bf16(&mut self) -> Result<(), String> {
        round_bf16(&mut self.weight);
        round_bf16(&mut self.bias);
        finite(&self.weight, "BF16 linear weight")?;
        finite(&self.bias, "BF16 linear bias")
    }
    pub fn forward(&self, x: &[f32]) -> Result<Vec<f32>, String> {
        if x.is_empty() || x.len() % self.input != 0 {
            return Err("Invalid Idol linear input shape".into());
        }
        let size = (x.len() / self.input)
            .checked_mul(self.output)
            .ok_or("Idol linear shape overflow")?;
        let mut y = vec![0.; size];
        y.par_chunks_mut(self.output)
            .zip(x.par_chunks(self.input))
            .for_each(|(out, row)| {
                for (i, v) in out.iter_mut().enumerate() {
                    *v = crate::ops::dot_f32(
                        row,
                        &self.weight[i * self.input..(i + 1) * self.input],
                        self.input,
                    ) + self.bias[i];
                }
            });
        Ok(y)
    }
}
pub(super) fn norm(
    x: &[f32],
    width: usize,
    weight: Option<&[f32]>,
    bias: Option<&[f32]>,
    eps: f32,
) -> Vec<f32> {
    assert!(width > 0 && x.len() % width == 0);
    let unit = vec![1.; width];
    let mut out = vec![0.; x.len()];
    out.par_chunks_mut(width)
        .zip(x.par_chunks(width))
        .for_each(|(o, r)| {
            crate::ops::layer_norm(r, weight.unwrap_or(&unit), bias.unwrap_or(&[]), eps, o)
        });
    out
}
pub(super) fn trace(name: &str, shape: &[usize], x: &[f32]) {
    #[cfg(feature = "parity-trace")]
    crate::parity_trace::report(crate::parity_trace::checkpoint(name, None, shape, x));
    let _ = (name, shape, x);
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_nonfinite_and_rounds_storage() {
        assert!(finite(&[f32::NAN], "input").is_err());
        let mut x = [1.0001];
        round_f16(&mut x);
        assert_eq!(x, [1.]);
    }
    #[test]
    fn reduced_precision_weight_overflow_is_rejected_during_load() {
        let mut linear = Linear {
            input: 1,
            output: 1,
            weight: vec![f32::MAX],
            bias: vec![0.],
        };
        assert!(linear.to_f16().is_err());
        linear.weight[0] = f32::MAX;
        assert!(linear.to_bf16().is_err());
    }
}
