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
        kind @ (GGMLType::Q8_0
        | GGMLType::Q4_0
        | GGMLType::Q4_1
        | GGMLType::Q4K
        | GGMLType::Q5K
        | GGMLType::Q6K) => {
            if shape.len() != 2 || shape.contains(&0) {
                return Err(format!(
                    "Idol block-quantized tensor {name} must be a matrix"
                ));
            }
            let count = usize::try_from(
                info.checked_n_elements()
                    .ok_or("Idol tensor size overflow")?,
            )
            .map_err(|_| "Idol tensor size does not fit usize")?;
            let columns = shape[1];
            let row_bytes = kind.nbytes(columns);
            let mut values = vec![0.; count];
            // Decode storage once; the existing graph retains its component precision boundaries.
            values
                .par_chunks_mut(columns)
                .zip(bytes.par_chunks_exact(row_bytes))
                .for_each(|(out, row)| {
                    if kind == GGMLType::Q4_1 {
                        crate::ops::embedding::embedding_lookup_q4_1(row, 0, columns, out);
                    } else {
                        crate::ops::embedding::embedding_lookup(row, 0, columns, kind, out);
                    }
                });
            values
        }
        _ => {
            return Err(format!(
                "Unsupported Idol tensor type {:?} for {name}",
                info.ggml_type
            ))
        }
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
pub(super) fn round_bf16_checked(x: &mut [f32], name: &str) -> Result<(), String> {
    round_bf16(x);
    finite(x, name)
}
/// Rounds activations to the reduced precision and rejects an overflow.
///
/// A value that the reduced precision cannot hold becomes an infinity on
/// rounding, and an infinity in an activation is not caught by a later
/// check on the same tensor: it is absorbed by the next reduction. The
/// rounding is therefore checked where it happens.
pub(super) fn round_f16_checked(x: &mut [f32], name: &str) -> Result<(), String> {
    round_f16(x);
    finite(x, name)
}
/// Rounds activations to the reduced precision and rejects an overflow.
pub(super) fn round_bf16_activation(x: &mut [f32], name: &str) -> Result<(), String> {
    round_bf16(x);
    finite(x, name)
}
pub(super) struct Linear {
    pub input: usize,
    pub output: usize,
    weight: Vec<f32>,
    bias: Vec<f32>,
    gpu: super::gpu::Projection,
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
            gpu: Default::default(),
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
        super::gpu::check_failure()?;
        if x.is_empty() || x.len() % self.input != 0 {
            return Err("Invalid Idol linear input shape".into());
        }
        let size = (x.len() / self.input)
            .checked_mul(self.output)
            .ok_or("Idol linear shape overflow")?;
        let mut y = vec![0.; size];
        if self
            .gpu
            .linear(&self.weight, x, &mut y, self.input, self.output)
        {
            for row in y.chunks_exact_mut(self.output) {
                for (value, bias) in row.iter_mut().zip(&self.bias) {
                    *value += bias;
                }
            }
            finite(&y, "GPU linear output")?;
            return Ok(y);
        }
        super::gpu::check_failure()?;
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
    struct WeightSource {
        info: crate::core::tensor::TensorInfo,
        data: Vec<u8>,
    }
    impl TensorSource for WeightSource {
        fn metadata(&self, _: &str) -> Option<&crate::core::tensor::MetaValue> {
            None
        }
        fn tensor_info(&self, name: &str) -> Option<&crate::core::tensor::TensorInfo> {
            (name == self.info.name).then_some(&self.info)
        }
        fn tensor_slice(&self, name: &str) -> Option<&[u8]> {
            (name == self.info.name).then_some(self.data.as_slice())
        }
    }
    #[test]
    fn gguf_block_weights_decode_and_reject_truncated_payloads() {
        for kind in [
            GGMLType::Q8_0,
            GGMLType::Q4_0,
            GGMLType::Q4_1,
            GGMLType::Q4K,
            GGMLType::Q5K,
            GGMLType::Q6K,
        ] {
            let info = crate::core::tensor::TensorInfo {
                name: "weight".into(),
                dims: vec![256, 2],
                ggml_type: kind,
                offset: 0,
            };
            let mut source = WeightSource {
                data: vec![0; info.checked_nbytes().unwrap() as usize],
                info,
            };
            assert_eq!(tensor(&source, "weight", &[2, 256]).unwrap(), vec![0.; 512]);
            source.data.pop();
            assert!(tensor(&source, "weight", &[2, 256]).is_err());
        }
    }
    #[test]
    fn gguf_q4_weight_rows_keep_low_then_high_nibble_order() {
        for kind in [GGMLType::Q4_0, GGMLType::Q4_1] {
            let mut data = Vec::new();
            for scale in [0.5f32, 2.] {
                data.extend(half::f16::from_f32(scale).to_bits().to_le_bytes());
                if kind == GGMLType::Q4_1 {
                    data.extend(half::f16::from_f32(-1.).to_bits().to_le_bytes());
                }
                data.extend([0xf0; 16]);
            }
            let source = WeightSource {
                info: crate::core::tensor::TensorInfo {
                    name: "weight".into(),
                    dims: vec![32, 2],
                    ggml_type: kind,
                    offset: 0,
                },
                data,
            };
            let values = tensor(&source, "weight", &[2, 32]).unwrap();
            for (row, scale) in [0.5f32, 2.].into_iter().enumerate() {
                let (low, high) = if kind == GGMLType::Q4_0 {
                    (-8. * scale, 7. * scale)
                } else {
                    (-1., 15. * scale - 1.)
                };
                assert_eq!(&values[row * 32..row * 32 + 16], &[low; 16]);
                assert_eq!(&values[row * 32 + 16..row * 32 + 32], &[high; 16]);
            }
        }
    }
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
            gpu: Default::default(),
        };
        assert!(linear.to_f16().is_err());
        linear.weight[0] = f32::MAX;
        assert!(linear.to_bf16().is_err());
    }
    #[test]
    fn bf16_parameter_overflow_cannot_become_silent_zero_normalization() {
        let mut variance = [f32::MAX];
        assert!(round_bf16_checked(&mut variance, "VAE variance").is_err());
        let mut ordinary = [1.0001];
        round_bf16_checked(&mut ordinary, "VAE variance").unwrap();
        assert_eq!(ordinary, [1.]);
    }
    #[test]
    fn activation_overflow_is_rejected_where_the_rounding_happens() {
        // The reduced precision cannot hold this value, and the rounding
        // turns it into an infinity, which the next reduction would absorb.
        let mut activation = [70_000.];
        assert!(round_f16_checked(&mut activation, "FP16 activation").is_err());
        let mut ordinary = [1.0001];
        round_f16_checked(&mut ordinary, "FP16 activation").unwrap();
        assert_eq!(ordinary, [1.]);
        // BF16 keeps the exponent range of F32, so a large finite value
        // survives the rounding; only an infinity is rejected.
        let mut large = [1.0e30];
        round_bf16_activation(&mut large, "BF16 activation").unwrap();
        assert!(large[0].is_finite());
        let mut infinite = [f32::INFINITY];
        assert!(round_bf16_activation(&mut infinite, "BF16 activation").is_err());
    }
}
