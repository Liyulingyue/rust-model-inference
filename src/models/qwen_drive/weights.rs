use crate::core::tensor::{GGMLType, TensorSource};
use crate::core::thread_pool::ComputePool;
use crate::ops::kernel::Weight;

enum HeadWeight<'a> {
    Planner(Weight<'a>),
    PerceptionBf16(Vec<u8>),
}

pub struct HeadLinear<'a> {
    weight: HeadWeight<'a>,
    bias: Option<Vec<f32>>,
    input: usize,
    output: usize,
}

#[derive(Default)]
pub struct HeadLinearScratch {
    rounded: Vec<u8>,
}

fn checked_matrix<'a, S: TensorSource + ?Sized>(
    source: &'a S,
    name: &str,
    input: usize,
    output: usize,
    expected_type: GGMLType,
) -> Result<&'a [u8], String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    let dims = [input as u64, output as u64];
    if info.dims != dims || info.ggml_type != expected_type {
        return Err(format!(
            "Invalid tensor {name}: shape {:?} type {:?}; expected {dims:?} {expected_type:?}",
            info.dims, info.ggml_type
        ));
    }
    let expected = usize::try_from(
        info.checked_nbytes()
            .ok_or_else(|| format!("Invalid tensor byte size: {name}"))?,
    )
    .map_err(|_| format!("Tensor byte size does not fit usize: {name}"))?;
    let bytes = source
        .tensor_slice(name)
        .ok_or_else(|| format!("Missing tensor data: {name}"))?;
    if bytes.len() != expected {
        return Err(format!(
            "Invalid tensor data length for {name}: {}; expected {expected}",
            bytes.len()
        ));
    }
    Ok(bytes)
}

fn load_bias<S: TensorSource + ?Sized>(
    source: &S,
    name: &str,
    output: usize,
    expected_type: GGMLType,
    round_bf16: bool,
) -> Result<Vec<f32>, String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    if info.dims != [output as u64] || info.ggml_type != expected_type {
        return Err(format!(
            "Invalid tensor {name}: shape {:?} type {:?}; expected [{output}] {expected_type:?}",
            info.dims, info.ggml_type
        ));
    }
    let bytes = source
        .tensor_slice(name)
        .ok_or_else(|| format!("Missing tensor data: {name}"))?;
    let expected = output
        .checked_mul(if expected_type == GGMLType::F32 { 4 } else { 2 })
        .ok_or_else(|| format!("Tensor byte size overflow: {name}"))?;
    if bytes.len() != expected {
        return Err(format!(
            "Invalid tensor data length for {name}: {}; expected {expected}",
            bytes.len()
        ));
    }
    let values: Vec<f32> = if expected_type == GGMLType::F32 {
        bytes
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte chunk")))
            .collect()
    } else {
        bytes
            .chunks_exact(2)
            .map(|bytes| crate::ops::bf16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]])))
            .collect()
    };
    if values.iter().any(|value| !value.is_finite()) {
        return Err(format!("Invalid finite tensor: {name}"));
    }
    if round_bf16 {
        Ok(values
            .into_iter()
            .map(|value| crate::ops::bf16_to_f32(crate::ops::f32_to_bf16(value)))
            .collect())
    } else {
        Ok(values)
    }
}

impl<'a> HeadLinear<'a> {
    pub fn load<S: TensorSource + ?Sized>(
        source: &'a S,
        name: &str,
        input: usize,
        output: usize,
        bias: bool,
    ) -> Result<Self, String> {
        let weight_name = format!("{name}.weight");
        checked_matrix(source, &weight_name, input, output, GGMLType::BF16)?;
        let weight = crate::models::qwen35::trunk::weights::load_weight(source, &weight_name)
            .ok_or_else(|| format!("Unsupported tensor: {weight_name}"))?;
        Ok(Self {
            weight: HeadWeight::Planner(weight),
            bias: bias
                .then(|| {
                    load_bias(
                        source,
                        &format!("{name}.bias"),
                        output,
                        GGMLType::BF16,
                        false,
                    )
                })
                .transpose()?,
            input,
            output,
        })
    }

    pub fn load_perception<S: TensorSource + ?Sized>(
        source: &'a S,
        name: &str,
        input: usize,
        output: usize,
        bias: bool,
    ) -> Result<Self, String> {
        let weight_name = format!("{name}.weight");
        let bytes = checked_matrix(source, &weight_name, input, output, GGMLType::F32)?;
        let mut weight = Vec::with_capacity(input * output * 2);
        for chunk in bytes.chunks_exact(4) {
            let value = f32::from_le_bytes(chunk.try_into().expect("four-byte chunk"));
            if !value.is_finite() {
                return Err(format!("Invalid finite tensor: {weight_name}"));
            }
            weight.extend_from_slice(&crate::ops::f32_to_bf16(value).to_le_bytes());
        }
        Ok(Self {
            weight: HeadWeight::PerceptionBf16(weight),
            bias: bias
                .then(|| load_bias(source, &format!("{name}.bias"), output, GGMLType::F32, true))
                .transpose()?,
            input,
            output,
        })
    }

    pub fn forward_one(&self, input: &[f32]) -> Result<Vec<f32>, String> {
        let mut output = vec![0.0; self.output];
        self.forward_rows(
            input,
            1,
            &ComputePool::new(1),
            &mut output,
            &mut HeadLinearScratch::default(),
        )?;
        Ok(output)
    }

    pub fn forward_rows(
        &self,
        input: &[f32],
        rows: usize,
        _pool: &ComputePool,
        output: &mut [f32],
        scratch: &mut HeadLinearScratch,
    ) -> Result<(), String> {
        if input.len() != rows.saturating_mul(self.input)
            || output.len() != rows.saturating_mul(self.output)
        {
            return Err(format!(
                "Qwen-Drive linear shape mismatch: input {}, output {}, rows {rows}, width {}x{}",
                input.len(),
                output.len(),
                self.input,
                self.output
            ));
        }
        for (input, output) in input
            .chunks_exact(self.input)
            .zip(output.chunks_exact_mut(self.output))
        {
            match &self.weight {
                HeadWeight::Planner(weight) => {
                    weight
                        .kernel
                        .forward(input, output, self.input, self.output);
                }
                HeadWeight::PerceptionBf16(weight) => {
                    scratch.rounded.clear();
                    scratch.rounded.extend(
                        input
                            .iter()
                            .flat_map(|value| crate::ops::f32_to_bf16(*value).to_le_bytes()),
                    );
                    for (row, value) in weight.chunks_exact(self.input * 2).zip(output.iter_mut()) {
                        *value = crate::ops::kernel::bf16::scalar::dot_bf16(row, &scratch.rounded);
                    }
                }
            }
            if let Some(bias) = &self.bias {
                for (value, bias) in output.iter_mut().zip(bias) {
                    *value += bias;
                }
            }
            if matches!(self.weight, HeadWeight::PerceptionBf16(_)) {
                for value in output {
                    *value = crate::ops::bf16_to_f32(crate::ops::f32_to_bf16(*value));
                }
            }
        }
        Ok(())
    }
}

pub struct HeadConv2d {
    pub(crate) weight: Vec<u8>,
    pub(crate) bias: Option<Vec<f32>>,
    pub(crate) input_channels: usize,
    pub(crate) output_channels: usize,
    pub(crate) kernel: [usize; 2],
}

impl HeadConv2d {
    pub fn load_perception<S: TensorSource + ?Sized>(
        source: &S,
        name: &str,
        input_channels: usize,
        output_channels: usize,
        kernel: [usize; 2],
        bias: bool,
    ) -> Result<Self, String> {
        let weight_name = format!("{name}.weight");
        let expected_dims = [
            kernel[1] as u64,
            kernel[0] as u64,
            input_channels as u64,
            output_channels as u64,
        ];
        let info = source
            .tensor_info(&weight_name)
            .ok_or_else(|| format!("Missing tensor: {weight_name}"))?;
        if info.dims != expected_dims || info.ggml_type != GGMLType::F32 {
            return Err(format!(
                "Invalid tensor {weight_name}: shape {:?} type {:?}; expected {expected_dims:?} F32",
                info.dims, info.ggml_type
            ));
        }
        let bytes = source
            .tensor_slice(&weight_name)
            .ok_or_else(|| format!("Missing tensor data: {weight_name}"))?;
        let elements = input_channels
            .checked_mul(output_channels)
            .and_then(|value| value.checked_mul(kernel[0]))
            .and_then(|value| value.checked_mul(kernel[1]))
            .ok_or_else(|| format!("Tensor byte size overflow: {weight_name}"))?;
        if bytes.len() != elements * 4 {
            return Err(format!("Invalid tensor data length for {weight_name}"));
        }
        let mut weight = Vec::with_capacity(elements * 2);
        for chunk in bytes.chunks_exact(4) {
            let value = f32::from_le_bytes(chunk.try_into().expect("four-byte chunk"));
            if !value.is_finite() {
                return Err(format!("Invalid finite tensor: {weight_name}"));
            }
            weight.extend_from_slice(&crate::ops::f32_to_bf16(value).to_le_bytes());
        }
        Ok(Self {
            weight,
            bias: bias
                .then(|| {
                    load_bias(
                        source,
                        &format!("{name}.bias"),
                        output_channels,
                        GGMLType::F32,
                        true,
                    )
                })
                .transpose()?,
            input_channels,
            output_channels,
            kernel,
        })
    }
}
