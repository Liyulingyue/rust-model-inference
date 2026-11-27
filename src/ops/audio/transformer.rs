//! Stateless audio transformer operations shared by audio encoders.
use crate::core::tensor::{load_f32_tensor, TensorSource};
use crate::ops::{dot_f32, gelu_erf, sum_f32, sum_sq_centered_f32};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::tensor::{GGMLType, MetaValue, TensorInfo};

    struct Source {
        info: [TensorInfo; 2],
        bytes: [Vec<u8>; 2],
    }
    impl TensorSource for Source {
        fn metadata(&self, _: &str) -> Option<&MetaValue> {
            None
        }
        fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
            self.info.iter().find(|i| i.name == name)
        }
        fn tensor_slice(&self, name: &str) -> Option<&[u8]> {
            self.info
                .iter()
                .position(|i| i.name == name)
                .map(|i| self.bytes[i].as_slice())
        }
    }

    #[test]
    fn layer_norm_load_preserves_decodable_float_formats_and_finite_checks() {
        for ty in [GGMLType::F32, GGMLType::BF16] {
            let encode = |values: &[f32]| -> Vec<u8> {
                values
                    .iter()
                    .flat_map(|&v| match ty {
                        GGMLType::F32 => v.to_le_bytes().to_vec(),
                        _ => crate::ops::f32_to_bf16(v).to_le_bytes().to_vec(),
                    })
                    .collect()
            };
            let mut s = Source {
                info: ["ln.weight", "ln.bias"].map(|name| TensorInfo {
                    name: name.into(),
                    dims: vec![2],
                    ggml_type: ty,
                    offset: 0,
                }),
                bytes: [encode(&[1.0, 0.5]), encode(&[0.25, -0.25])],
            };
            let loaded = LayerNormWeights::load(&s, "ln", &[2]).unwrap();
            assert_eq!(loaded.weight, [1.0, 0.5]);
            assert_eq!(loaded.bias, [0.25, -0.25]);
            s.bytes[0] = encode(&[f32::INFINITY, 0.5]);
            assert!(LayerNormWeights::load(&s, "ln", &[2]).is_err());
            s.bytes[0].pop();
            assert!(LayerNormWeights::load(&s, "ln", &[2]).is_err());
        }
    }
}

pub(crate) struct LayerNormWeights {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
}

impl LayerNormWeights {
    pub(crate) fn load(
        source: &dyn TensorSource,
        prefix: &str,
        dims: &[u64],
    ) -> Result<Self, String> {
        let load = |suffix: &str| -> Result<Vec<f32>, String> {
            let name = format!("{prefix}.{suffix}");
            let values = load_f32_tensor(source, &name, dims)?;
            if values.is_empty() || values.iter().any(|v| !v.is_finite()) {
                return Err(format!("Invalid finite layer norm tensor: {name}"));
            }
            Ok(values)
        };
        Ok(Self {
            weight: load("weight")?,
            bias: load("bias")?,
        })
    }
}

pub(crate) fn layer_norm(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    epsilon: f32,
    output: &mut [f32],
) -> Result<(), String> {
    if input.is_empty()
        || input.len() != weight.len()
        || input.len() != bias.len()
        || input.len() != output.len()
        || !epsilon.is_finite()
        || epsilon < 0.0
        || input
            .iter()
            .chain(weight)
            .chain(bias)
            .any(|value| !value.is_finite())
    {
        return Err("Invalid layer norm tensors".into());
    }

    let n = input.len();
    let sum = sum_f32(input);
    let mean = (sum / n as f64) as f32;
    let variance = (sum_sq_centered_f32(input, mean) / n as f64) as f32;
    let inverse_std = 1.0 / (variance + epsilon).sqrt();
    if !inverse_std.is_finite() {
        return Err("Non-finite layer norm statistics".into());
    }
    for ((output, input), (weight, bias)) in
        output.iter_mut().zip(input).zip(weight.iter().zip(bias))
    {
        *output = (*input - mean) * inverse_std * weight + bias;
    }
    if output.iter().any(|value| !value.is_finite()) {
        return Err("Non-finite layer norm output".into());
    }
    Ok(())
}

/// Apply layer normalization to `rows` independent rows.
pub(crate) fn layer_norm_rows(
    input: &[f32],
    rows: usize,
    weights: &LayerNormWeights,
    epsilon: f32,
    output: &mut Vec<f32>,
) -> Result<(), String> {
    if rows == 0 || weights.weight.is_empty() || weights.weight.len() != weights.bias.len() {
        return Err("Invalid audio layer norm weights".into());
    }
    let width = weights.weight.len();
    let len = checked_product("audio layer norm values", rows, width)?;
    if input.len() != len {
        return Err("Invalid audio layer norm input".into());
    }
    resize_f32(output, "audio layer norm output", len)?;
    for row in 0..rows {
        layer_norm(
            &input[row * width..(row + 1) * width],
            &weights.weight,
            &weights.bias,
            epsilon,
            &mut output[row * width..(row + 1) * width],
        )?;
    }
    Ok(())
}

pub(crate) fn apply_gelu_erf(values: &mut [f32]) -> Result<(), String> {
    if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
        return Err("Invalid audio GELU input".into());
    }
    for value in values {
        *value = gelu_erf(*value);
        if !value.is_finite() {
            return Err("Non-finite audio GELU output".into());
        }
    }
    Ok(())
}

pub(crate) fn attention_softmax(scores: &mut [f32]) -> Result<(), String> {
    crate::ops::softmax_inplace(scores);
    if scores.iter().any(|score| !score.is_finite()) {
        return Err("Invalid attention softmax".into());
    }
    Ok(())
}

pub(crate) fn transpose_value_head(
    value: &[f32],
    tokens: usize,
    width: usize,
    head: usize,
    head_dim: usize,
    columns: &mut [f32],
) {
    debug_assert_eq!(value.len(), tokens * width);
    debug_assert_eq!(columns.len(), tokens * head_dim);
    debug_assert!((head + 1) * head_dim <= width);
    let head_start = head * head_dim;
    for dimension in 0..head_dim {
        let column = &mut columns[dimension * tokens..(dimension + 1) * tokens];
        for (key_token, target) in column.iter_mut().enumerate() {
            *target = value[key_token * width + head_start + dimension];
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn full_attention_into(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    tokens: usize,
    heads: usize,
    head_dim: usize,
    scores: &mut Vec<f32>,
    output: &mut Vec<f32>,
) -> Result<(), String> {
    if tokens == 0 || heads == 0 || head_dim == 0 {
        return Err("Attention dimensions must be non-zero".into());
    }
    let width = checked_product("attention width", heads, head_dim)?;
    let len = checked_product("attention values", tokens, width)?;
    if query.len() != len
        || key.len() != len
        || value.len() != len
        || query
            .iter()
            .chain(key)
            .chain(value)
            .any(|value| !value.is_finite())
    {
        return Err("Invalid full attention tensors".into());
    }
    resize_f32(scores, "attention scores", tokens)?;
    resize_f32(output, "attention output", len)?;
    let value_columns_len = checked_product("attention value columns", tokens, head_dim)?;
    let mut value_columns = reserved_f32("attention value columns", value_columns_len)?;
    output.fill(0.0);
    let scale = 1.0 / (head_dim as f32).sqrt();
    for head in 0..heads {
        transpose_value_head(value, tokens, width, head, head_dim, &mut value_columns);
        for query_token in 0..tokens {
            let query_start = query_token * width + head * head_dim;
            let query_head = &query[query_start..query_start + head_dim];
            for key_token in 0..tokens {
                let key_start = key_token * width + head * head_dim;
                let dot = dot_f32(query_head, &key[key_start..key_start + head_dim], head_dim);
                let score = dot * scale;
                if !score.is_finite() {
                    return Err("Non-finite attention score".into());
                }
                scores[key_token] = score;
            }
            attention_softmax(&mut scores[..tokens])?;
            let output_start = query_token * width + head * head_dim;
            let output_row = &mut output[output_start..output_start + head_dim];
            for dimension in 0..head_dim {
                let column = &value_columns[dimension * tokens..(dimension + 1) * tokens];
                output_row[dimension] = dot_f32(column, scores, tokens);
            }
        }
    }
    if output.iter().any(|value| !value.is_finite()) {
        return Err("Non-finite attention output".into());
    }
    Ok(())
}

pub(crate) fn checked_product(name: &str, left: usize, right: usize) -> Result<usize, String> {
    left.checked_mul(right)
        .ok_or_else(|| format!("{name} overflows usize"))
}

pub(crate) fn reserved_f32(name: &str, len: usize) -> Result<Vec<f32>, String> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| format!("Failed to allocate {name}"))?;
    values.resize(len, 0.0);
    Ok(values)
}

pub(crate) fn resize_f32(values: &mut Vec<f32>, name: &str, len: usize) -> Result<(), String> {
    if len > values.len() {
        values
            .try_reserve_exact(len - values.len())
            .map_err(|_| format!("Failed to allocate {name}"))?;
    }
    values.resize(len, 0.0);
    Ok(())
}
