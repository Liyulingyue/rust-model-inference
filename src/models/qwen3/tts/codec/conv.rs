//! 1D convolution primitives for the Qwen3-TTS codec decoder.
//!
//! The codec decoder uses three variants:
//!
//! - [`conv1d`]: standard causal cross-correlation. Used by DAC residual blocks.
//! - [`conv_transpose1d`]: transposed convolution (a.k.a. fractionally-strided
//!   convolution) used to upsample in DAC upsampling blocks.
//!
//! All buffers follow the same `[in_channels, length]` / `[out_channels, length]`
//! row-major convention as [`snake1d_inplace`](crate::models::qwen3::tts::codec::snake::snake1d_inplace).

use rayon::prelude::*;

#[cfg(target_arch = "aarch64")]
use crate::ops::dot::dot_f32_neon;
use crate::ops::{dot_f16, f32_to_f16};

/// Standard 1D convolution (no padding, full output).
///
/// `kernel` has shape `[out_channels, in_channels, kernel_size]` (PyTorch
/// convention for `Conv1d` weights). `input` and `output` are stored as
/// `[in_channels, length_in]` and `[out_channels, length_out]` respectively
/// where `length_out = length_in - kernel_size + 1`.
///
/// The output is summed with `bias` if provided.
pub fn conv1d(
    kernel: &[f32],
    bias: Option<&[f32]>,
    input: &[f32],
    in_channels: usize,
    length_in: usize,
    out_channels: usize,
    kernel_size: usize,
    length_out: usize,
) -> Result<Vec<f32>, String> {
    let expected_in = in_channels * length_in;
    let expected_kernel = out_channels * in_channels * kernel_size;
    if input.len() != expected_in {
        return Err(format!(
            "conv1d: input length {} != expected {}",
            input.len(),
            expected_in,
        ));
    }
    if kernel.len() != expected_kernel {
        return Err(format!(
            "conv1d: kernel length {} != expected {}",
            kernel.len(),
            expected_kernel,
        ));
    }
    if let Some(bias) = bias {
        if bias.len() != out_channels {
            return Err(format!(
                "conv1d: bias length {} != out_channels {}",
                bias.len(),
                out_channels,
            ));
        }
    }
    let mut output = vec![0.0f32; out_channels * length_out];
    for oc in 0..out_channels {
        let bias_val = bias.map_or(0.0, |bias| bias[oc]);
        let base = oc * length_out;
        for o in 0..length_out {
            let mut acc = bias_val;
            for ic in 0..in_channels {
                for k in 0..kernel_size {
                    let kernel_idx = oc * in_channels * kernel_size + ic * kernel_size + k;
                    let input_idx = ic * length_in + (o + k);
                    acc += kernel[kernel_idx] * input[input_idx];
                }
            }
            output[base + o] = acc;
        }
    }
    Ok(output)
}

/// Transposed 1D convolution with stride = `stride` and zero padding.
///
/// `kernel` has shape `[in_channels, out_channels, kernel_size]` (PyTorch
/// convention for `ConvTranspose1d` weights). `input` is `[in_channels,
/// length_in]`; the output is `[out_channels, length_out]` where
/// `length_out = (length_in - 1) * stride + kernel_size`.
///
/// The output is summed with `bias` if provided.
pub fn conv_transpose1d(
    kernel: &[f32],
    bias: Option<&[f32]>,
    input: &[f32],
    in_channels: usize,
    length_in: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
) -> Result<(usize, Vec<f32>), String> {
    let expected_in = in_channels * length_in;
    let expected_kernel = in_channels * out_channels * kernel_size;
    if input.len() != expected_in {
        return Err(format!(
            "conv_transpose1d: input length {} != expected {}",
            input.len(),
            expected_in,
        ));
    }
    if kernel.len() != expected_kernel {
        return Err(format!(
            "conv_transpose1d: kernel length {} != expected {}",
            kernel.len(),
            expected_kernel,
        ));
    }
    if let Some(bias) = bias {
        if bias.len() != out_channels {
            return Err(format!(
                "conv_transpose1d: bias length {} != out_channels {}",
                bias.len(),
                out_channels,
            ));
        }
    }
    let length_out = (length_in.saturating_sub(1))
        .checked_mul(stride)
        .and_then(|v| v.checked_add(kernel_size))
        .ok_or_else(|| "conv_transpose1d: output length overflow".to_string())?;
    let mut output = vec![0.0f32; out_channels * length_out];
    if let Some(bias) = bias {
        for oc in 0..out_channels {
            let base = oc * length_out;
            for v in output[base..base + length_out].iter_mut() {
                *v = bias[oc];
            }
        }
    }
    for ic in 0..in_channels {
        for oc in 0..out_channels {
            for i in 0..length_in {
                for k in 0..kernel_size {
                    let out_idx = i * stride + k;
                    if out_idx >= length_out {
                        continue;
                    }
                    let kernel_idx = ic * out_channels * kernel_size + oc * kernel_size + k;
                    let input_idx = ic * length_in + i;
                    output[oc * length_out + out_idx] += kernel[kernel_idx] * input[input_idx];
                }
            }
        }
    }
    Ok((length_out, output))
}

#[derive(Debug, Default)]
pub struct CausalConv1dState {
    context: Vec<f32>,
    channels: usize,
    pad: usize,
}

#[derive(Debug, Default)]
pub struct ConvTranspose1dState {
    tail: Vec<f32>,
    channels: usize,
    overlap: usize,
}

/// Stateful causal Conv1d over T-first `[length, channels]` buffers. GGUF
/// kernels use `[kernel, in_channels, out_channels]` layout.
pub fn conv1d_causal(
    kernel: &[u16],
    bias: Option<&[f32]>,
    input: &[f32],
    in_channels: usize,
    length_in: usize,
    out_channels: usize,
    kernel_size: usize,
    dilation: usize,
    state: &mut CausalConv1dState,
) -> Result<Vec<f32>, String> {
    if in_channels == 0 || out_channels == 0 || kernel_size == 0 || dilation == 0 {
        return Err("conv1d_causal: dimensions and dilation must be nonzero".into());
    }
    let expected_input = in_channels
        .checked_mul(length_in)
        .ok_or_else(|| "conv1d_causal: input size overflow".to_string())?;
    let expected_kernel = kernel_size
        .checked_mul(in_channels)
        .and_then(|size| size.checked_mul(out_channels))
        .ok_or_else(|| "conv1d_causal: kernel size overflow".to_string())?;
    if input.len() != expected_input || kernel.len() != expected_kernel {
        return Err(format!(
            "conv1d_causal: input/kernel length {}/{} != expected {expected_input}/{expected_kernel}",
            input.len(),
            kernel.len()
        ));
    }
    if bias.is_some_and(|values| values.len() != out_channels) {
        return Err("conv1d_causal: bias length mismatch".into());
    }
    let pad = (kernel_size - 1)
        .checked_mul(dilation)
        .ok_or_else(|| "conv1d_causal: padding overflow".to_string())?;
    if pad > 0 {
        if state.context.is_empty() {
            state.context = vec![0.0; pad * in_channels];
            state.channels = in_channels;
            state.pad = pad;
        } else if state.channels != in_channels || state.pad != pad {
            return Err("conv1d_causal: state shape changed within a request".into());
        }
    }
    let full_len = pad
        .checked_add(length_in)
        .ok_or_else(|| "conv1d_causal: padded length overflow".to_string())?;
    let mut full = Vec::with_capacity(full_len * in_channels);
    full.extend_from_slice(&state.context);
    full.extend_from_slice(input);
    let mut output = vec![0.0; length_in * out_channels];
    output
        .par_chunks_mut(out_channels)
        .enumerate()
        .for_each(|(time, row)| {
            let dot_len = in_channels * kernel_size;
            let mut input_f16 = vec![0; dot_len];
            for kernel_index in 0..kernel_size {
                let input_time = time + kernel_index * dilation;
                let input_row = &full[input_time * in_channels..(input_time + 1) * in_channels];
                for input_channel in 0..in_channels {
                    input_f16[input_channel * kernel_size + kernel_index] =
                        f32_to_f16(input_row[input_channel]);
                }
            }
            for (output_channel, value) in row.iter_mut().enumerate() {
                let weights = &kernel[output_channel * dot_len..(output_channel + 1) * dot_len];
                *value = dot_f16(weights, &input_f16, dot_len)
                    + bias.map_or(0.0, |values| values[output_channel]);
            }
        });
    if pad > 0 {
        state
            .context
            .copy_from_slice(&full[length_in * in_channels..]);
    }
    Ok(output)
}

/// Depthwise counterpart for GGUF `[kernel, 1, channels]` weights.
pub fn conv1d_causal_depthwise(
    kernel: &[u16],
    bias: Option<&[f32]>,
    input: &[f32],
    channels: usize,
    length_in: usize,
    kernel_size: usize,
    state: &mut CausalConv1dState,
) -> Result<Vec<f32>, String> {
    if kernel.len() != kernel_size * channels || input.len() != length_in * channels {
        return Err("conv1d_causal_depthwise: input/kernel length mismatch".into());
    }
    if bias.is_some_and(|values| values.len() != channels) {
        return Err("conv1d_causal_depthwise: bias length mismatch".into());
    }
    let pad = kernel_size.saturating_sub(1);
    if pad > 0 {
        if state.context.is_empty() {
            state.context = vec![0.0; pad * channels];
            state.channels = channels;
            state.pad = pad;
        } else if state.channels != channels || state.pad != pad {
            return Err("conv1d_causal_depthwise: state shape changed within a request".into());
        }
    }
    let mut full = Vec::with_capacity((pad + length_in) * channels);
    full.extend_from_slice(&state.context);
    full.extend_from_slice(input);
    let mut output = vec![0.0; length_in * channels];
    for time in 0..length_in {
        let mut input_f16 = vec![0; kernel_size];
        for channel in 0..channels {
            for kernel_index in 0..kernel_size {
                input_f16[kernel_index] =
                    f32_to_f16(full[(time + kernel_index) * channels + channel]);
            }
            let weights = &kernel[channel * kernel_size..(channel + 1) * kernel_size];
            output[time * channels + channel] = dot_f16(weights, &input_f16, kernel_size)
                + bias.map_or(0.0, |values| values[channel]);
        }
    }
    if pad > 0 {
        state.context.copy_from_slice(&full[length_in * channels..]);
    }
    Ok(output)
}

/// Stateful causal ConvTranspose1d over T-first buffers. GGUF kernels use
/// `[kernel, out_channels, in_channels]` layout.
pub fn conv_transpose1d_causal(
    kernel: &[f32],
    bias: Option<&[f32]>,
    input: &[f32],
    in_channels: usize,
    length_in: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    state: &mut ConvTranspose1dState,
) -> Result<Vec<f32>, String> {
    if in_channels == 0
        || out_channels == 0
        || kernel_size == 0
        || stride == 0
        || kernel_size < stride
    {
        return Err("conv_transpose1d_causal: invalid dimensions or stride".into());
    }
    if input.len() != length_in * in_channels
        || kernel.len() != kernel_size * out_channels * in_channels
    {
        return Err("conv_transpose1d_causal: input/kernel length mismatch".into());
    }
    if bias.is_some_and(|values| values.len() != out_channels) {
        return Err("conv_transpose1d_causal: bias length mismatch".into());
    }
    let overlap = kernel_size - stride;
    if overlap > 0 {
        if state.tail.is_empty() {
            state.tail = vec![0.0; overlap * out_channels];
            state.channels = out_channels;
            state.overlap = overlap;
        } else if state.channels != out_channels || state.overlap != overlap {
            return Err("conv_transpose1d_causal: state shape changed within a request".into());
        }
    }
    let emit_len = length_in
        .checked_mul(stride)
        .ok_or_else(|| "conv_transpose1d_causal: emitted length overflow".to_string())?;
    let full_len = emit_len
        .checked_add(overlap)
        .ok_or_else(|| "conv_transpose1d_causal: output length overflow".to_string())?;
    let mut transposed_kernel = vec![0.0; kernel.len()];
    for input_channel in 0..in_channels {
        for output_channel in 0..out_channels {
            for kernel_index in 0..kernel_size {
                transposed_kernel
                    [(output_channel * kernel_size + kernel_index) * in_channels + input_channel] =
                    kernel[(input_channel * out_channels + output_channel) * kernel_size
                        + kernel_index];
            }
        }
    }
    let mut full = vec![0.0; full_len * out_channels];
    full.par_chunks_mut(out_channels)
        .enumerate()
        .for_each(|(output_time, row)| {
            if length_in == 0 {
                return;
            }
            let first_input = output_time
                .saturating_add(1)
                .saturating_sub(kernel_size)
                .div_ceil(stride);
            let last_input = (output_time / stride).min(length_in - 1);
            for input_time in first_input..=last_input {
                let kernel_index = output_time - input_time * stride;
                let input_row = &input[input_time * in_channels..(input_time + 1) * in_channels];
                for (output_channel, value) in row.iter_mut().enumerate() {
                    let start = (output_channel * kernel_size + kernel_index) * in_channels;
                    let weights = &transposed_kernel[start..start + in_channels];
                    #[cfg(target_arch = "aarch64")]
                    let dot = unsafe { dot_f32_neon(input_row, weights, in_channels) };
                    #[cfg(not(target_arch = "aarch64"))]
                    let dot = input_row
                        .iter()
                        .zip(weights)
                        .fold(0.0f32, |sum, (&input, &weight)| sum + input * weight);
                    *value += dot;
                }
            }
        });
    for (value, tail) in full.iter_mut().zip(&state.tail) {
        *value += *tail;
    }
    let mut output = full[..emit_len * out_channels].to_vec();
    if let Some(bias) = bias {
        for row in output.chunks_exact_mut(out_channels) {
            for (value, bias) in row.iter_mut().zip(bias) {
                *value += *bias;
            }
        }
    }
    if overlap > 0 {
        state.tail.copy_from_slice(&full[emit_len * out_channels..]);
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn causal_conv_matches_when_input_is_split() {
        let kernel = [0.25, 0.5, 1.0].map(f32_to_f16);
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut whole_state = CausalConv1dState::default();
        let whole = conv1d_causal(&kernel, None, &input, 1, 6, 1, 3, 1, &mut whole_state).unwrap();
        let mut split_state = CausalConv1dState::default();
        let mut split =
            conv1d_causal(&kernel, None, &input[..2], 1, 2, 1, 3, 1, &mut split_state).unwrap();
        split.extend(
            conv1d_causal(&kernel, None, &input[2..], 1, 4, 1, 3, 1, &mut split_state).unwrap(),
        );
        assert_eq!(
            split
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            whole
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn causal_conv_uses_ggml_kernel_first_layout() {
        let kernel = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0].map(f32_to_f16);
        let mut state = CausalConv1dState::default();
        let output = conv1d_causal(&kernel, None, &[1.0, 10.0], 2, 1, 2, 2, 1, &mut state).unwrap();
        assert_eq!(output, [42.0, 86.0]);
    }

    #[test]
    fn causal_transpose_conv_overlap_add_matches_whole_input() {
        let kernel = [1.0, 2.0, 3.0, 4.0];
        let input = [1.0, 2.0, 3.0];
        let mut whole_state = ConvTranspose1dState::default();
        let whole = conv_transpose1d_causal(&kernel, None, &input, 1, 3, 1, 4, 2, &mut whole_state)
            .unwrap();
        let mut split_state = ConvTranspose1dState::default();
        let mut split =
            conv_transpose1d_causal(&kernel, None, &input[..1], 1, 1, 1, 4, 2, &mut split_state)
                .unwrap();
        split.extend(
            conv_transpose1d_causal(&kernel, None, &input[1..], 1, 2, 1, 4, 2, &mut split_state)
                .unwrap(),
        );
        assert_eq!(
            split
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            whole
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }
}
