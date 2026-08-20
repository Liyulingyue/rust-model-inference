//! 4-stage DAC upsampler (Descript Audio Codec) for the Qwen3-TTS codec.
//!
//! The DAC decoder upsamples a `[512, length]` waveform embedding to a single
//! audio channel at `length * 16 * 10 * 8 * 6 = length * 7680` samples via
//! four ConvTranspose1d upsampling blocks:
//!
//! | Block | In ch | Out ch | Stride |
//! |-------|-------|--------|--------|
//! | entry | 1024  | 1536   | 1      |
//! | up0   | 1536  | 768    | 16     |
//! | up1   | 768   | 384    | 10     |
//! | up2   | 384   | 192    | 8      |
//! | up3   | 192   | 96     | 6      |
//! | post  | 96    | 1      | 7      |
//!
//! Each upsample block is `Snake1d -> ConvTranspose1d -> 3 residual blocks`.

use crate::core::tensor::{GGMLType, TensorSource};
use crate::models::qwen3::{checked_product, load_f32_tensor, usize_to_u64};
use crate::models::tts::codec::conv::{conv1d, conv_transpose1d};
use crate::models::tts::codec::snake::snake1d_inplace;

const DAC_ENTRY_KERNEL: usize = 7;
const DAC_POST_KERNEL: usize = 7;
const DAC_UP_STRIDES: [usize; 4] = [16, 10, 8, 6];
const DAC_RESIDUAL_KERNELS: [usize; 4] = [7, 7, 7, 3];

pub struct DacBlock {
    snake_alpha: Vec<f32>,
    snake_beta: Vec<f32>,
    conv_weight: Vec<f32>,
    conv_bias: Vec<f32>,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    residual: Vec<DacResidual>,
}

pub struct DacResidual {
    act1_alpha: Vec<f32>,
    act1_beta: Vec<f32>,
    conv1_weight: Vec<f32>,
    conv1_bias: Vec<f32>,
    act2_alpha: Vec<f32>,
    act2_beta: Vec<f32>,
    conv2_weight: Vec<f32>,
    conv2_bias: Vec<f32>,
}

pub struct DacDecoder {
    entry_weight: Vec<f32>,
    entry_bias: Vec<f32>,
    blocks: Vec<DacBlock>,
    post_snake_alpha: Vec<f32>,
    post_snake_beta: Vec<f32>,
    post_weight: Vec<f32>,
    post_bias: Vec<f32>,
}

impl DacDecoder {
    pub fn from_source(source: &dyn TensorSource) -> Result<Self, String> {
        let entry_weight = load_f16_or_f32_tensor(
            source,
            "a.gen.wav.dac.entry.weight",
            &[DAC_ENTRY_KERNEL as u64, 1024, 1536],
        )?;
        let entry_bias = load_f32_tensor(
            source,
            "a.gen.wav.dac.entry.bias",
            &[usize_to_u64(1536, "dac entry out")?],
        )?;
        let post_snake_alpha = load_f32_tensor(
            source,
            "a.gen.wav.dac.post_snake.alpha",
            &[usize_to_u64(96, "post snake alpha")?],
        )?;
        let post_snake_beta = load_f32_tensor(
            source,
            "a.gen.wav.dac.post_snake.beta",
            &[usize_to_u64(96, "post snake beta")?],
        )?;
        let post_weight = load_f16_or_f32_tensor(
            source,
            "a.gen.wav.dac.post_conv.weight",
            &[DAC_POST_KERNEL as u64, 96, 1],
        )?;
        let post_bias = load_f32_tensor(
            source,
            "a.gen.wav.dac.post_conv.bias",
            &[1],
        )?;

        let mut blocks = Vec::with_capacity(DAC_UP_STRIDES.len());
        for (block_idx, stride) in DAC_UP_STRIDES.iter().enumerate() {
            let block = load_dac_block(source, block_idx, *stride)?;
            blocks.push(block);
        }

        Ok(Self {
            entry_weight,
            entry_bias,
            blocks,
            post_snake_alpha,
            post_snake_beta,
            post_weight,
            post_bias,
        })
    }

    /// Decode `[in_channels, length]` waveform embedding to `[1, audio_length]`.
    pub fn decode(&self, input: &[f32], in_channels: usize, length: usize) -> Result<Vec<f32>, String> {
        if input.len() != in_channels * length {
            return Err(format!(
                "DacDecoder: input length {} != in_channels*length {}",
                input.len(),
                in_channels * length,
            ));
        }
        if in_channels != 1024 {
            return Err(format!(
                "DacDecoder: expected 1024 input channels, got {in_channels}",
            ));
        }

        // Entry: Conv1d 1024 -> 1536 with kernel 7 (length shrinks by kernel_size-1).
        let (mut current_len, mut current) = conv1d_rearranged(
            &self.entry_weight,
            Some(&self.entry_bias),
            input,
            1024,
            length,
            1536,
            DAC_ENTRY_KERNEL,
        )?;

        for block in &self.blocks {
            current = block.forward(&current, current_len)?;
            current_len = current.len() / block.out_channels;
        }
        let block_last_out_channels = self.blocks.last().unwrap().out_channels;
        let block_last_len = current.len() / block_last_out_channels;

        // Post: Snake -> Conv1d 96 -> 1 with kernel 7.
        snake1d_inplace(&mut current, block_last_len, &self.post_snake_alpha, &self.post_snake_beta)?;
        let (post_len, post_out) = conv1d_rearranged(
            &self.post_weight,
            Some(&self.post_bias),
            &current,
            block_last_out_channels,
            block_last_len,
            1,
            DAC_POST_KERNEL,
        )?;
        if post_out.is_empty() {
            return Err("DacDecoder: post-conv produced empty output".into());
        }
        // post_out is [1, post_len]; flatten to [post_len].
        Ok(post_out)
    }
}

impl DacBlock {
    fn forward(&self, input: &[f32], length: usize) -> Result<Vec<f32>, String> {
        if input.len() != self.in_channels * length {
            return Err(format!(
                "DacBlock.forward: input length {} != expected {}",
                input.len(),
                self.in_channels * length,
            ));
        }
        snake1d_inplace(input.to_owned().as_mut_slice(), length, &self.snake_alpha, &self.snake_beta)?;
        // Snake1d modifies the input — but we own a copy after to_owned().
        // Wait, we want to operate on the input directly without copying. Let's
        // pass a temporary owned buffer and snake it in-place.
        let mut snake_input = input.to_vec();
        snake1d_inplace(&mut snake_input, length, &self.snake_alpha, &self.snake_beta)?;

        let (up_len, upsampled) = conv_transpose1d(
            &self.conv_weight,
            Some(&self.conv_bias),
            &snake_input,
            self.in_channels,
            length,
            self.out_channels,
            self.kernel_size,
            self.stride,
        )?;
        // Apply residuals.
        let mut current = upsampled;
        let mut current_len = up_len;
        for residual in &self.residual {
            current = residual.forward(&current, current_len)?;
            // residual.forward keeps the same length.
        }
        Ok(current)
    }
}

impl DacResidual {
    fn forward(&self, input: &[f32], length: usize) -> Result<Vec<f32>, String> {
        let ch = self.act1_alpha.len();
        if input.len() != ch * length {
            return Err(format!(
                "DacResidual.forward: input length {} != expected {}",
                input.len(),
                ch * length,
            ));
        }
        // Snake1d (act1).
        let mut after_act1 = input.to_vec();
        snake1d_inplace(&mut after_act1, length, &self.act1_alpha, &self.act1_beta)?;
        // Conv1d ch -> ch, kernel 7/3 (same length).
        let (after_conv1_len, after_conv1) = conv1d_rearranged(
            &self.conv1_weight,
            Some(&self.conv1_bias),
            &after_act1,
            ch,
            length,
            ch,
            self.conv1_kernel_size(),
        )?;
        // Snake1d (act2).
        let mut after_act2 = after_conv1;
        snake1d_inplace(&mut after_act2, after_conv1_len, &self.act2_alpha, &self.act2_beta)?;
        // Conv1d ch -> ch, kernel 1 (same length).
        let (after_conv2_len, after_conv2) = conv1d_rearranged(
            &self.conv2_weight,
            Some(&self.conv2_bias),
            &after_act2,
            ch,
            after_conv1_len,
            ch,
            self.conv2_kernel_size(),
        )?;
        // Residual add.
        let mut out = input.to_vec();
        for (acc, add) in out.iter_mut().zip(after_conv2.iter()) {
            *acc += *add;
        }
        debug_assert_eq!(out.len(), ch * after_conv2_len);
        Ok(out)
    }

    fn conv1_kernel_size(&self) -> usize {
        // kernel = conv1_weight total / (ch * ch) — assumes (ch, ch, k)
        self.conv1_weight.len() / (self.act1_alpha.len() * self.act1_alpha.len())
    }

    fn conv2_kernel_size(&self) -> usize {
        self.conv2_weight.len() / (self.act1_alpha.len() * self.act1_alpha.len())
    }
}

fn load_dac_block(
    source: &dyn TensorSource,
    block_idx: usize,
    stride: usize,
) -> Result<DacBlock, String> {
    let prefix = format!("a.gen.wav.dac.blk.{block_idx}");
    let (in_channels, out_channels, kernel_size) = dac_block_channels(block_idx)?;
    let snake_alpha = load_f32_tensor(
        source,
        &format!("{prefix}.snake.alpha"),
        &[usize_to_u64(in_channels, "dac block snake alpha")?],
    )?;
    let snake_beta = load_f32_tensor(
        source,
        &format!("{prefix}.snake.beta"),
        &[usize_to_u64(in_channels, "dac block snake beta")?],
    )?;
let conv_weight = load_f16_or_f32_tensor(
            source,
            &format!("{prefix}.conv.weight"),
            &[kernel_size as u64, out_channels as u64, in_channels as u64],
        )?;
    let conv_bias = load_f32_tensor(
        source,
        &format!("{prefix}.conv.bias"),
        &[usize_to_u64(out_channels, "dac block conv bias")?],
    )?;
    let n_residuals = 3;
    let mut residual = Vec::with_capacity(n_residuals);
    for r in 0..n_residuals {
        let res_prefix = format!("{prefix}.res.{r}");
        let res_kernel = dac_residual_conv1_kernel(block_idx);
        let res_kernel2 = 1;
        let act1_alpha = load_f32_tensor(
            source,
            &format!("{res_prefix}.act1.alpha"),
            &[usize_to_u64(out_channels, "dac res act1 alpha")?],
        )?;
        let act1_beta = load_f32_tensor(
            source,
            &format!("{res_prefix}.act1.beta"),
            &[usize_to_u64(out_channels, "dac res act1 beta")?],
        )?;
        let conv1_weight = load_f16_or_f32_tensor(
            source,
            &format!("{res_prefix}.conv1.weight"),
            &[res_kernel as u64, out_channels as u64, out_channels as u64],
        )?;
        let conv1_bias = load_f32_tensor(
            source,
            &format!("{res_prefix}.conv1.bias"),
            &[usize_to_u64(out_channels, "dac res conv1 bias")?],
        )?;
        let act2_alpha = load_f32_tensor(
            source,
            &format!("{res_prefix}.act2.alpha"),
            &[usize_to_u64(out_channels, "dac res act2 alpha")?],
        )?;
        let act2_beta = load_f32_tensor(
            source,
            &format!("{res_prefix}.act2.beta"),
            &[usize_to_u64(out_channels, "dac res act2 beta")?],
        )?;
        let conv2_weight = load_f16_or_f32_tensor(
            source,
            &format!("{res_prefix}.conv2.weight"),
            &[res_kernel2 as u64, out_channels as u64, out_channels as u64],
        )?;
        let conv2_bias = load_f32_tensor(
            source,
            &format!("{res_prefix}.conv2.bias"),
            &[usize_to_u64(out_channels, "dac res conv2 bias")?],
        )?;
        residual.push(DacResidual {
            act1_alpha,
            act1_beta,
            conv1_weight,
            conv1_bias,
            act2_alpha,
            act2_beta,
            conv2_weight,
            conv2_bias,
        });
    }

    Ok(DacBlock {
        snake_alpha,
        snake_beta,
        conv_weight,
        conv_bias,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        residual,
    })
}

fn dac_block_channels(block_idx: usize) -> Result<(usize, usize, usize), String> {
    // (in_channels, out_channels, kernel_size) per block.
    match block_idx {
        0 => Ok((1536, 768, 16)),
        1 => Ok((768, 384, 10)),
        2 => Ok((384, 192, 8)),
        3 => Ok((192, 96, 6)),
        _ => Err(format!("unknown DAC block index {block_idx}")),
    }
}

fn dac_residual_conv1_kernel(block_idx: usize) -> usize {
    match block_idx {
        0 | 1 | 2 => DAC_RESIDUAL_KERNELS[block_idx],
        3 => 7,
        _ => 7,
    }
}

/// Run `conv1d` on a kernel stored in the GGUF layout `[k, in, out]` (the
/// reverse of PyTorch's `[out, in, k]`) and return the resulting
/// `[out_channels, length_out]` buffer along with its length.
fn conv1d_rearranged(
    kernel: &[f32],
    bias: Option<&[f32]>,
    input: &[f32],
    in_channels: usize,
    length_in: usize,
    out_channels: usize,
    kernel_size: usize,
) -> Result<(usize, Vec<f32>), String> {
    let expected_kernel = in_channels * out_channels * kernel_size;
    if kernel.len() != expected_kernel {
        return Err(format!(
            "conv1d_rearranged: kernel length {} != expected {}",
            kernel.len(),
            expected_kernel,
        ));
    }
    let length_out = length_in
        .checked_sub(kernel_size)
        .ok_or_else(|| "conv1d_rearranged: kernel larger than input".to_string())?
        .checked_add(1)
        .ok_or_else(|| "conv1d_rearranged: length_out overflow".to_string())?;
    let mut rearranged = vec![0.0f32; expected_kernel];
    // GGUF layout: [k, in, out]. Permute to PyTorch Conv1d [out, in, k].
    for k in 0..kernel_size {
        for ic in 0..in_channels {
            for oc in 0..out_channels {
                let src_idx = k * in_channels * out_channels + ic * out_channels + oc;
                let dst_idx = oc * in_channels * kernel_size + ic * kernel_size + k;
                rearranged[dst_idx] = kernel[src_idx];
            }
        }
    }
    conv1d(
        &rearranged,
        bias,
        input,
        in_channels,
        length_in,
        out_channels,
        kernel_size,
        length_out,
    )
    .map(|v| (length_out, v))
}

fn load_f16_or_f32_tensor(
    source: &dyn TensorSource,
    name: &str,
    expected_dims: &[u64],
) -> Result<Vec<f32>, String> {
    let bytes = source
        .tensor_slice(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing tensor info: {name}"))?;
    if info.dims != expected_dims {
        return Err(format!(
            "{name}: dims {:?} != expected {expected_dims:?}",
            info.dims,
        ));
    }
    if info.ggml_type != GGMLType::F16 && info.ggml_type != GGMLType::F32 {
        return Err(format!(
            "{name}: type {:?} not F16 or F32",
            info.ggml_type,
        ));
    }
    match info.ggml_type {
        GGMLType::F16 => Ok(bytes
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect()),
        GGMLType::F32 => Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()),
        _ => unreachable!(),
    }
}