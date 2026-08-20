//! 1D convolution primitives for the Qwen3-TTS codec decoder.
//!
//! The codec decoder uses three variants:
//!
//! - [`conv1d`]: standard causal cross-correlation. Used by DAC residual blocks.
//! - [`conv_transpose1d`]: transposed convolution (a.k.a. fractionally-strided
//!   convolution) used to upsample in DAC upsampling blocks.
//!
//! All buffers follow the same `[in_channels, length]` / `[out_channels, length]`
//! row-major convention as [`snake1d_inplace`](crate::models::tts::codec::snake::snake1d_inplace).

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
                    output[oc * length_out + out_idx] +=
                        kernel[kernel_idx] * input[input_idx];
                }
            }
        }
    }
    Ok((length_out, output))
}