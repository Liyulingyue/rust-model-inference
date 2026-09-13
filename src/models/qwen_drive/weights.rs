use crate::core::tensor::{GGMLType, TensorSource};
use crate::core::thread_pool::ComputePool;

enum HeadWeight<'a> {
    Planner(&'a [u8]),
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

#[inline(always)]
fn bf16_at(bytes: &[u8], index: usize) -> f32 {
    let offset = index * 2;
    crate::ops::bf16_to_f32(u16::from_le_bytes([bytes[offset], bytes[offset + 1]]))
}

// PyTorch 2.8's AArch64 BF16 GEMV widens eight BF16 values into two F32x4
// accumulators, keeps eight accumulators per 32-value block, then reduces them
// as a fixed tree. The order is observable when the result is rounded to BF16.
fn torch28_bf16_dot(weight: &[u8], input: &[u8]) -> f32 {
    debug_assert_eq!(weight.len(), input.len());
    let len = weight.len() / 2;
    let aligned = len & !31;
    let mut sums = [[0.0f32; 4]; 8];
    for base in (0..aligned).step_by(32) {
        for (register, sum) in sums.iter_mut().enumerate() {
            for (lane, value) in sum.iter_mut().enumerate() {
                let index = base + register * 4 + lane;
                *value = bf16_at(weight, index).mul_add(bf16_at(input, index), *value);
            }
        }
    }
    for offset in [4, 2, 1] {
        for register in 0..offset {
            for lane in 0..4 {
                sums[register][lane] += sums[register + offset][lane];
            }
        }
    }
    let mut result = (sums[0][0] + sums[0][1]) + (sums[0][2] + sums[0][3]);

    let vector_aligned = len & !7;
    let mut tail = [0.0f32; 4];
    for base in (aligned..vector_aligned).step_by(8) {
        for lane in 0..4 {
            tail[lane] =
                bf16_at(weight, base + lane).mul_add(bf16_at(input, base + lane), tail[lane]);
            tail[lane] = bf16_at(weight, base + 4 + lane)
                .mul_add(bf16_at(input, base + 4 + lane), tail[lane]);
        }
    }
    result += (tail[0] + tail[1]) + (tail[2] + tail[3]);
    for index in vector_aligned..len {
        result += bf16_at(weight, index) * bf16_at(input, index);
    }
    result
}

impl<'a> HeadLinear<'a> {
    pub(crate) fn output(&self) -> usize {
        self.output
    }

    pub fn load<S: TensorSource + ?Sized>(
        source: &'a S,
        name: &str,
        input: usize,
        output: usize,
        bias: bool,
    ) -> Result<Self, String> {
        let weight_name = format!("{name}.weight");
        let weight = checked_matrix(source, &weight_name, input, output, GGMLType::BF16)?;
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

    pub fn load_perception_slice<S: TensorSource + ?Sized>(
        source: &'a S,
        name: &str,
        input: usize,
        full_output: usize,
        rows: std::ops::Range<usize>,
    ) -> Result<Self, String> {
        if rows.is_empty() || rows.end > full_output {
            return Err(format!("Invalid Qwen-Drive linear row range: {rows:?}"));
        }
        let weight_name = format!("{name}_weight");
        let bytes = checked_matrix(source, &weight_name, input, full_output, GGMLType::F32)?;
        let row_bytes = input * 4;
        let mut weight = Vec::with_capacity((rows.end - rows.start) * input * 2);
        for chunk in bytes[rows.start * row_bytes..rows.end * row_bytes].chunks_exact(4) {
            let value = f32::from_le_bytes(chunk.try_into().expect("four-byte chunk"));
            if !value.is_finite() {
                return Err(format!("Invalid finite tensor: {weight_name}"));
            }
            weight.extend_from_slice(&crate::ops::f32_to_bf16(value).to_le_bytes());
        }
        let bias = load_bias(
            source,
            &format!("{name}_bias"),
            full_output,
            GGMLType::F32,
            true,
        )?[rows.clone()]
        .to_vec();
        Ok(Self {
            weight: HeadWeight::PerceptionBf16(weight),
            bias: Some(bias),
            input,
            output: rows.end - rows.start,
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
        pool: &ComputePool,
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
        match &self.weight {
            HeadWeight::Planner(weight) => {
                scratch.rounded.clear();
                scratch.rounded.extend(
                    input
                        .iter()
                        .flat_map(|value| crate::ops::f32_to_bf16(*value).to_le_bytes()),
                );
                let input_ptr = scratch.rounded.as_ptr() as usize;
                let weight_ptr = weight.as_ptr() as usize;
                let output_ptr = output.as_mut_ptr() as usize;
                pool.compute(|thread, threads| {
                    for task in (thread..rows * self.output).step_by(threads) {
                        let row = task / self.output;
                        let output_index = task % self.output;
                        let input_offset = row * self.input * 2;
                        let weight_offset = output_index * self.input * 2;
                        let input = unsafe {
                            std::slice::from_raw_parts(
                                (input_ptr as *const u8).add(input_offset),
                                self.input * 2,
                            )
                        };
                        let weight = unsafe {
                            std::slice::from_raw_parts(
                                (weight_ptr as *const u8).add(weight_offset),
                                self.input * 2,
                            )
                        };
                        let sum = torch28_bf16_dot(weight, input);
                        unsafe { *(output_ptr as *mut f32).add(task) = sum };
                    }
                });
            }
            HeadWeight::PerceptionBf16(weight) => {
                for (input, output) in input
                    .chunks_exact(self.input)
                    .zip(output.chunks_exact_mut(self.output))
                {
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
        }
        for output in output.chunks_exact_mut(self.output) {
            if let Some(bias) = &self.bias {
                for (value, bias) in output.iter_mut().zip(bias) {
                    *value += bias;
                }
            }
            for value in output {
                *value = crate::ops::bf16_to_f32(crate::ops::f32_to_bf16(*value));
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planner_linear_is_bitwise_stable_across_thread_counts() {
        const INPUT: usize = 64;
        const OUTPUT: usize = 17;
        const ROWS: usize = 3;
        let mut state = 42_u32;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            f32::from_bits(0x3f00_0000 | (state & 0x007f_ffff)) - 1.0
        };
        let weight = (0..INPUT * OUTPUT)
            .flat_map(|_| crate::ops::f32_to_bf16(next()).to_le_bytes())
            .collect::<Vec<_>>();
        let bias = (0..OUTPUT).map(|_| next()).collect::<Vec<_>>();
        let input = (0..ROWS * INPUT).map(|_| next()).collect::<Vec<_>>();
        let linear = HeadLinear {
            weight: HeadWeight::Planner(&weight),
            bias: Some(bias),
            input: INPUT,
            output: OUTPUT,
        };
        let mut single = vec![0.0; ROWS * OUTPUT];
        let mut parallel = vec![0.0; ROWS * OUTPUT];
        linear
            .forward_rows(
                &input,
                ROWS,
                &ComputePool::new(1),
                &mut single,
                &mut HeadLinearScratch::default(),
            )
            .unwrap();
        linear
            .forward_rows(
                &input,
                ROWS,
                &ComputePool::new(12),
                &mut parallel,
                &mut HeadLinearScratch::default(),
            )
            .unwrap();

        assert_eq!(
            single
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            parallel
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }
}
