//! Validated mmap-backed matrices shared by dots.tts components.
use crate::core::tensor::{GGMLType, TensorSource};
use crate::ops::kernel::{QuantizedTensor, Weight};

pub(crate) fn load_weight<'a>(
    source: &'a dyn TensorSource,
    name: &str,
    dims: &[u64],
) -> Result<Weight<'a>, String> {
    if dims.len() < 2 || dims.contains(&0) {
        return Err(format!("Invalid matrix dimensions for {name}: {dims:?}"));
    }
    let n_out = usize::try_from(*dims.last().unwrap())
        .map_err(|_| format!("{name} output width overflow"))?;
    let n_in = dims[..dims.len() - 1]
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(usize::try_from(d).ok()?))
        .ok_or_else(|| format!("{name} input width overflow"))?;
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    let flattened = [n_in as u64, n_out as u64];
    if info.dims != dims && !(info.ggml_type == GGMLType::Q8_0 && info.dims == flattened) {
        return Err(format!(
            "Invalid tensor {name}: shape {:?}; expected {dims:?}",
            info.dims
        ));
    }
    if !matches!(
        info.ggml_type,
        GGMLType::F32 | GGMLType::F16 | GGMLType::BF16 | GGMLType::Q8_0
    ) || (info.ggml_type == GGMLType::Q8_0 && n_in % 32 != 0)
    {
        return Err(format!(
            "Unsupported matrix {name}: {:?}, row width {n_in}",
            info.ggml_type
        ));
    }
    let expected = n_in
        .checked_mul(n_out)
        .and_then(|count| match info.ggml_type {
            GGMLType::F32 => count.checked_mul(4),
            GGMLType::F16 | GGMLType::BF16 => count.checked_mul(2),
            GGMLType::Q8_0 => (count / 32).checked_mul(34),
            _ => None,
        })
        .ok_or_else(|| format!("Tensor byte size overflow: {name}"))?;
    let bytes = source
        .tensor_slice(name)
        .ok_or_else(|| format!("Missing tensor data: {name}"))?;
    if bytes.len() != expected {
        return Err(format!(
            "Invalid tensor data length for {name}: {}; expected {expected}",
            bytes.len()
        ));
    }
    let mut weight = Weight::from_quantized(QuantizedTensor::from_bytes(
        bytes,
        info.ggml_type,
        n_in,
        n_out,
    ));
    // The existing F32 enum does not retain matrix shape.
    weight.n_out = n_out;
    weight.n_in = n_in;
    Ok(weight)
}

pub(crate) fn linear_forward(
    weight: &Weight<'_>,
    bias: Option<&[f32]>,
    input: &[f32],
    in_dim: usize,
    out_dim: usize,
    output: &mut [f32],
) {
    assert_eq!((weight.n_in, weight.n_out), (in_dim, out_dim));
    assert_eq!(input.len() % in_dim, 0);
    assert_eq!(output.len(), input.len() / in_dim * out_dim);
    if let Some(values) = weight.kernel.f32_slice() {
        for (input, output) in input
            .chunks_exact(in_dim)
            .zip(output.chunks_exact_mut(out_dim))
        {
            for (row, value) in values.chunks_exact(in_dim).zip(output) {
                *value = crate::ops::dot_f32(row, input, in_dim);
            }
        }
    } else {
        weight
            .kernel
            .forward_batched(input, output, in_dim, out_dim);
    }
    if let Some(bias) = bias {
        assert_eq!(bias.len(), out_dim);
        for row in output.chunks_exact_mut(out_dim) {
            for (value, bias) in row.iter_mut().zip(bias) {
                *value += bias;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::tensor::{GGMLType, MetaValue, TensorInfo};
    struct Source {
        info: TensorInfo,
        bytes: Vec<u8>,
    }
    impl TensorSource for Source {
        fn metadata(&self, _: &str) -> Option<&MetaValue> {
            None
        }
        fn tensor_info(&self, _: &str) -> Option<&TensorInfo> {
            Some(&self.info)
        }
        fn tensor_slice(&self, _: &str) -> Option<&[u8]> {
            Some(&self.bytes)
        }
    }
    #[test]
    fn flattened_q8_convolution_loads_without_dequantizing_and_runs_batched() {
        let mut bytes = Vec::new();
        for value in [1i8, -2] {
            bytes.extend_from_slice(&0x3c00u16.to_le_bytes());
            bytes.extend([value as u8; 32]);
        }
        let mut source = Source {
            info: TensorInfo {
                name: "conv.weight".into(),
                dims: vec![32, 2],
                ggml_type: GGMLType::Q8_0,
                offset: 0,
            },
            bytes,
        };
        {
            let weight = load_weight(&source, "conv.weight", &[2, 16, 2]).unwrap();
            assert_eq!(weight.ggml_type, GGMLType::Q8_0);
            // amax/127 is exactly 1 or 2, avoiding activation scale rounding.
            let input = [vec![127.0; 32], vec![254.0; 32]].concat();
            let mut output = [0.0; 4];
            linear_forward(&weight, Some(&[3.0, 4.0]), &input, 32, 2, &mut output);
            assert_eq!(output, [4067.0, -8124.0, 8131.0, -16252.0]);
        }
        source.bytes.pop();
        assert!(load_weight(&source, "conv.weight", &[2, 16, 2]).is_err());
        assert!(load_weight(&source, "conv.weight", &[31, 2]).is_err());
        assert!(load_weight(&source, "conv.weight", &[u64::MAX, 2, 2]).is_err());
    }
}
