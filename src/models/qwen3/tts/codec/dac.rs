//! 4-stage DAC upsampler (Descript Audio Codec) plus 2 ConvNeXt upsample
//! blocks for the Qwen3-TTS codec.
//!
//! Full decode pipeline (T-first layout, `[length, channels]`):
//!
//! 1. `pre_conv` (Conv1d k=3): 512 → 1024 channels.
//! 2. **2 upsample stages** (each): causal ConvTranspose1d (kernel=2, stride=2)
//!    followed by a ConvNeXt block (dwconv k=7 → LayerNorm → pwconv1 4× → GELU
//!    → pwconv2 → per-channel gamma).
//! 3. `dac_entry`: Conv1d k=7, 1024 → 1536.
//! 4. **4 DAC blocks**: Snake → causal ConvTranspose1d (kernel=2×stride, stride
//!    in {8, 5, 4, 3}) → 3 residual units with dilations `[1, 3, 9]`.
//! 5. `dac_post`: Snake → Conv1d k=7, 96 → 1 channel.
//! Clamping is deferred to PCM serialization.

use crate::core::tensor::{GGMLType, TensorSource};
use crate::models::qwen3::qwen3_multimodal::{load_f32_tensor, static_q8_matrix, usize_to_u64};
use crate::models::qwen3::tts::codec::conv::{
    conv1d_causal, conv1d_causal_depthwise, conv_transpose1d_causal, CausalConv1dState,
    ConvTranspose1dState,
};
use crate::models::qwen3::tts::codec::snake::snake1d_inplace;
use crate::models::qwen3::tts::{load_f16_or_f32_tensor, load_f16_tensor};
use crate::ops::{f16_to_f32, f32_to_f16, matmul_q8_0_quantized_parallel, quantize_q8_0_into};

#[cfg(unix)]
#[cfg_attr(not(target_vendor = "apple"), link(name = "m"))]
unsafe extern "C" {
    fn tanhf(value: f32) -> f32;
}

const DAC_ENTRY_KERNEL: usize = 7;
const DAC_POST_KERNEL: usize = 7;
const DAC_BLOCK_KERNELS: [usize; 4] = [16, 10, 8, 6];
const DAC_BLOCK_STRIDES: [usize; 4] = [8, 5, 4, 3];
const DAC_DILATIONS: [usize; 3] = [1, 3, 9];
const UPSAMPLE_KERNEL: usize = 2;
const UPSAMPLE_STRIDE: usize = 2;
const CONVNEXT_KERNEL: usize = 7;

pub struct DacDecoder {
    /// Causal Conv1d 512 → 1024, kernel 3.
    pre_conv_w: Vec<u16>,
    pre_conv_b: Vec<f32>,
    /// 2 ConvNeXt upsample stages (each 2× upsampling).
    upsample_blocks: Vec<UpsampleBlock>,
    /// Conv1d 1024 → 1536, kernel 7.
    entry_weight: Vec<u16>,
    entry_bias: Vec<f32>,
    /// 4 Snake + ConvTranspose1d + 3-residual-unit blocks.
    blocks: Vec<DacBlock>,
    /// Snake + Conv1d 96 → 1, kernel 7.
    post_snake_alpha: Vec<f32>,
    post_snake_beta: Vec<f32>,
    post_weight: Vec<u16>,
    post_bias: Vec<f32>,
}

pub struct UpsampleBlock {
    /// Causal ConvTranspose1d k=2, stride=2 (channel-preserving).
    conv_w: Vec<f32>,
    conv_b: Vec<f32>,
    /// Depthwise Conv1d k=7 (per-channel, single group).
    dwconv_w: Vec<u16>,
    dwconv_b: Vec<f32>,
    /// LayerNorm over channel dim.
    norm_w: Vec<f32>,
    norm_b: Vec<f32>,
    /// Pointwise Conv1d: pw1 (1024 → 4096), pw2 (4096 → 1024).
    pw1_w: Vec<u8>,
    pw1_b: Vec<f32>,
    pw2_w: Vec<u8>,
    pw2_b: Vec<f32>,
    /// Per-channel scale (gamma) for the residual branch.
    gamma: Vec<f32>,
}

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
    conv1_weight: Vec<u16>,
    conv1_bias: Vec<f32>,
    dilation: usize,
    act2_alpha: Vec<f32>,
    act2_beta: Vec<f32>,
    conv2_weight: Vec<u16>,
    conv2_bias: Vec<f32>,
}

#[derive(Debug, Default)]
pub struct DacState {
    pre_conv: CausalConv1dState,
    upsample: [ConvTranspose1dState; 2],
    upsample_depthwise: [CausalConv1dState; 2],
    entry: CausalConv1dState,
    block_upsample: [ConvTranspose1dState; 4],
    residual: [[CausalConv1dState; 3]; 4],
    post_conv: CausalConv1dState,
}

impl DacState {
    pub fn new() -> Self {
        Self::default()
    }
}

impl DacDecoder {
    pub fn from_source(source: &dyn TensorSource) -> Result<Self, String> {
        Self::from_source_dyn(source)
    }
}

impl DacDecoder {
    fn from_source_dyn(source: &dyn TensorSource) -> Result<Self, String> {
        let pre_conv_w = load_f16_tensor(source, "a.gen.wav.pre_conv.weight", &[3, 512, 1024])?;
        let pre_conv_b = load_f32_tensor(
            source,
            "a.gen.wav.pre_conv.bias",
            &[usize_to_u64(1024, "pre_conv bias")?],
        )?;
        // The reference decoder expects 1024-dim input to the DAC entry conv,
        // so the DAC's `decode` method starts at 1024-dim. The 512 → 1024
        // pre_conv is exposed separately (via [`Self::pre_conv`]) for callers
        // that have 512-dim RVQ output (the common case).
        let entry_weight = load_f16_tensor(
            source,
            "a.gen.wav.dac.entry.weight",
            &[DAC_ENTRY_KERNEL as u64, 1024, 1536],
        )?;
        let entry_bias = load_f32_tensor(
            source,
            "a.gen.wav.dac.entry.bias",
            &[usize_to_u64(1536, "dac entry bias")?],
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
        let post_weight = load_f16_tensor(
            source,
            "a.gen.wav.dac.post_conv.weight",
            &[DAC_POST_KERNEL as u64, 96, 1],
        )?;
        let post_bias = load_f32_tensor(source, "a.gen.wav.dac.post_conv.bias", &[1])?;

        // 2 upsample blocks.
        let mut upsample_blocks = Vec::with_capacity(2);
        for block_idx in 0..2 {
            upsample_blocks.push(load_upsample_block(source, block_idx)?);
        }

        // 4 DAC blocks.
        let mut blocks = Vec::with_capacity(DAC_BLOCK_STRIDES.len());
        for (block_idx, stride) in DAC_BLOCK_STRIDES.iter().enumerate() {
            blocks.push(load_dac_block(source, block_idx, *stride)?);
        }

        Ok(Self {
            pre_conv_w,
            pre_conv_b,
            upsample_blocks,
            entry_weight,
            entry_bias,
            blocks,
            post_snake_alpha,
            post_snake_beta,
            post_weight,
            post_bias,
        })
    }

    /// Apply the 512 → 1024 causal pre-convolution to a T-first buffer.
    pub fn pre_conv(&self, input: &[f32], length: usize) -> Result<Vec<f32>, String> {
        let mut state = DacState::new();
        self.pre_conv_window(input, length, &mut state)
    }

    pub fn pre_conv_window(
        &self,
        input: &[f32],
        length: usize,
        state: &mut DacState,
    ) -> Result<Vec<f32>, String> {
        if input.len() != 512 * length {
            return Err(format!(
                "DacDecoder.pre_conv: input length {} != 512 * length {}",
                input.len(),
                length,
            ));
        }
        let output = conv1d_causal(
            &self.pre_conv_w,
            Some(&self.pre_conv_b),
            input,
            512,
            length,
            1024,
            3,
            1,
            &mut state.pre_conv,
        )?;
        #[cfg(feature = "parity-trace")]
        crate::parity_trace::report(crate::parity_trace::checkpoint(
            "tts.wav_pre_conv",
            None,
            &[length, 1024],
            &output,
        ));
        Ok(output)
    }

    /// Decode a T-first `[length, 1024]` waveform embedding into mono F32 PCM.
    pub fn decode(&self, input: &[f32], length: usize) -> Result<Vec<f32>, String> {
        let mut state = DacState::new();
        self.decode_window(input, length, &mut state)
    }

    pub fn decode_window(
        &self,
        input: &[f32],
        length: usize,
        state: &mut DacState,
    ) -> Result<Vec<f32>, String> {
        if input.len() != 1024 * length {
            return Err(format!(
                "DacDecoder: input length {} != 1024 * length {}",
                input.len(),
                length,
            ));
        }

        let mut current = input.to_vec();
        let mut current_len = length;
        for (block_idx, block) in self.upsample_blocks.iter().enumerate() {
            current = upsample_block_forward(
                block,
                &current,
                current_len,
                &mut state.upsample[block_idx],
                &mut state.upsample_depthwise[block_idx],
            )?;
            current_len = current.len() / 1024;
        }

        let mut cur = conv1d_causal(
            &self.entry_weight,
            Some(&self.entry_bias),
            &current,
            1024,
            current_len,
            1536,
            DAC_ENTRY_KERNEL,
            1,
            &mut state.entry,
        )?;
        let mut cur_len = current_len;

        for (block_index, block) in self.blocks.iter().enumerate() {
            cur = block.forward_window(
                &cur,
                cur_len,
                &mut state.block_upsample[block_index],
                &mut state.residual[block_index],
            )?;
            cur_len = cur.len() / block.out_channels;
        }
        let block_last_out = self.blocks.last().unwrap().out_channels;
        let block_last_len = cur.len() / block_last_out;

        snake1d_inplace(
            &mut cur,
            block_last_len,
            &self.post_snake_alpha,
            &self.post_snake_beta,
        )?;
        let output = conv1d_causal(
            &self.post_weight,
            Some(&self.post_bias),
            &cur,
            block_last_out,
            block_last_len,
            1,
            DAC_POST_KERNEL,
            1,
            &mut state.post_conv,
        )?;
        #[cfg(feature = "parity-trace")]
        crate::parity_trace::report(crate::parity_trace::checkpoint(
            "tts.pcm",
            None,
            &[output.len()],
            &output,
        ));
        Ok(output)
    }
}

impl DacBlock {
    fn forward_window(
        &self,
        input: &[f32],
        length: usize,
        upsample_state: &mut ConvTranspose1dState,
        residual_states: &mut [CausalConv1dState; 3],
    ) -> Result<Vec<f32>, String> {
        if input.len() != self.in_channels * length {
            return Err(format!(
                "DacBlock.forward: input length {} != expected {}",
                input.len(),
                self.in_channels * length,
            ));
        }
        let mut after_snake = input.to_vec();
        snake1d_inplace(
            &mut after_snake,
            length,
            &self.snake_alpha,
            &self.snake_beta,
        )?;
        let mut cur = conv_transpose1d_causal(
            &self.conv_weight,
            Some(&self.conv_bias),
            &after_snake,
            self.in_channels,
            length,
            self.out_channels,
            self.kernel_size,
            self.stride,
            upsample_state,
        )?;
        let cur_len = length * self.stride;
        for (residual, residual_state) in self.residual.iter().zip(residual_states) {
            cur = residual.forward_window(&cur, cur_len, residual_state)?;
        }
        Ok(cur)
    }
}

impl DacResidual {
    fn forward_window(
        &self,
        input: &[f32],
        length: usize,
        state: &mut CausalConv1dState,
    ) -> Result<Vec<f32>, String> {
        let ch = self.act1_alpha.len();
        if input.len() != ch * length {
            return Err(format!(
                "DacResidual.forward: input length {} != expected {}",
                input.len(),
                ch * length,
            ));
        }
        let mut after_act1 = input.to_vec();
        snake1d_inplace(&mut after_act1, length, &self.act1_alpha, &self.act1_beta)?;
        let kernel_size = self.conv1_weight.len() / (ch * ch);
        let after_conv1 = conv1d_causal(
            &self.conv1_weight,
            Some(&self.conv1_bias),
            &after_act1,
            ch,
            length,
            ch,
            kernel_size,
            self.dilation,
            state,
        )?;
        let mut after_act2 = after_conv1;
        snake1d_inplace(&mut after_act2, length, &self.act2_alpha, &self.act2_beta)?;
        let after_conv2 = conv1d_causal(
            &self.conv2_weight,
            Some(&self.conv2_bias),
            &after_act2,
            ch,
            length,
            ch,
            1,
            1,
            &mut CausalConv1dState::default(),
        )?;
        let mut out = input.to_vec();
        for (acc, add) in out.iter_mut().zip(after_conv2.iter()) {
            *acc += *add;
        }
        Ok(out)
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
        &[usize_to_u64(in_channels, "dac snake alpha")?],
    )?;
    let snake_beta = load_f32_tensor(
        source,
        &format!("{prefix}.snake.beta"),
        &[usize_to_u64(in_channels, "dac snake beta")?],
    )?;
    let conv_weight = load_f16_or_f32_tensor(
        source,
        &format!("{prefix}.conv.weight"),
        &[kernel_size as u64, out_channels as u64, in_channels as u64],
    )?;
    let conv_bias = load_f32_tensor(
        source,
        &format!("{prefix}.conv.bias"),
        &[usize_to_u64(out_channels, "dac conv bias")?],
    )?;
    let mut residual = Vec::with_capacity(DAC_DILATIONS.len());
    for r in 0..DAC_DILATIONS.len() {
        let res_prefix = format!("{prefix}.res.{r}");
        let res_kernel = dac_residual_conv1_kernel(block_idx, r);
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
        let conv1_weight = load_f16_tensor(
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
        let conv2_weight = load_f16_tensor(
            source,
            &format!("{res_prefix}.conv2.weight"),
            &[1, out_channels as u64, out_channels as u64],
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
            dilation: DAC_DILATIONS[r],
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

fn load_upsample_block(
    source: &dyn TensorSource,
    block_idx: usize,
) -> Result<UpsampleBlock, String> {
    let prefix = format!("a.gen.wav.up.blk.{block_idx}");
    let conv_w = load_f32_tensor_3d(
        source,
        &format!("{prefix}.conv.weight"),
        UPSAMPLE_KERNEL,
        1024,
        1024,
    )?;
    let conv_b = load_f32_tensor(
        source,
        &format!("{prefix}.conv.bias"),
        &[usize_to_u64(1024, "upsample conv bias")?],
    )?;
    let dwconv_w = load_f16_tensor(
        source,
        &format!("{prefix}.dwconv.weight"),
        &[CONVNEXT_KERNEL as u64, 1, 1024],
    )?;
    let dwconv_b = load_f32_tensor(
        source,
        &format!("{prefix}.dwconv.bias"),
        &[usize_to_u64(1024, "upsample dwconv bias")?],
    )?;
    let norm_w = load_f32_tensor(
        source,
        &format!("{prefix}.norm.weight"),
        &[usize_to_u64(1024, "upsample norm weight")?],
    )?;
    let norm_b = load_f32_tensor(
        source,
        &format!("{prefix}.norm.bias"),
        &[usize_to_u64(1024, "upsample norm bias")?],
    )?;
    let pw1_w = static_q8_matrix(source, &format!("{prefix}.pw1.weight"), 1024, 4096)?.to_vec();
    let pw1_b = load_f32_tensor(
        source,
        &format!("{prefix}.pw1.bias"),
        &[usize_to_u64(4096, "upsample pw1 bias")?],
    )?;
    let pw2_w = static_q8_matrix(source, &format!("{prefix}.pw2.weight"), 4096, 1024)?.to_vec();
    let pw2_b = load_f32_tensor(
        source,
        &format!("{prefix}.pw2.bias"),
        &[usize_to_u64(1024, "upsample pw2 bias")?],
    )?;
    let gamma = load_f32_tensor(
        source,
        &format!("{prefix}.gamma"),
        &[usize_to_u64(1024, "upsample gamma")?],
    )?;
    Ok(UpsampleBlock {
        conv_w,
        conv_b,
        dwconv_w,
        dwconv_b,
        norm_w,
        norm_b,
        pw1_w,
        pw1_b,
        pw2_w,
        pw2_b,
        gamma,
    })
}

fn upsample_block_forward(
    block: &UpsampleBlock,
    input: &[f32],
    length: usize,
    upsample_state: &mut ConvTranspose1dState,
    depthwise_state: &mut CausalConv1dState,
) -> Result<Vec<f32>, String> {
    let upsampled = conv_transpose1d_causal(
        &block.conv_w,
        Some(&block.conv_b),
        input,
        1024,
        length,
        1024,
        UPSAMPLE_KERNEL,
        UPSAMPLE_STRIDE,
        upsample_state,
    )?;
    let upsampled_len = length * UPSAMPLE_STRIDE;
    let dwconv_out = conv1d_causal_depthwise(
        &block.dwconv_w,
        Some(&block.dwconv_b),
        &upsampled,
        1024,
        upsampled_len,
        CONVNEXT_KERNEL,
        depthwise_state,
    )?;
    let mut normalized = vec![0.0f32; dwconv_out.len()];
    for (row, output) in dwconv_out
        .chunks_exact(1024)
        .zip(normalized.chunks_exact_mut(1024))
    {
        let mean = row
            .iter()
            .fold(0.0f64, |sum, &value| sum + f64::from(value)) as f32
            / 1024.0;
        let mut variance = 0.0f64;
        #[cfg(target_arch = "aarch64")]
        unsafe {
            use std::arch::aarch64::*;
            let mean = vdupq_n_f32(mean);
            for channel in (0..1024).step_by(4) {
                let centered = vsubq_f32(vld1q_f32(row.as_ptr().add(channel)), mean);
                vst1q_f32(output.as_mut_ptr().add(channel), centered);
                variance += f64::from(vaddvq_f32(vmulq_f32(centered, centered)));
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        for channel in (0..1024).step_by(4) {
            let centered = [
                row[channel] - mean,
                row[channel + 1] - mean,
                row[channel + 2] - mean,
                row[channel + 3] - mean,
            ];
            output[channel..channel + 4].copy_from_slice(&centered);
            variance += f64::from(
                (centered[0] * centered[0] + centered[1] * centered[1])
                    + (centered[2] * centered[2] + centered[3] * centered[3]),
            );
        }
        let scale = 1.0 / ((variance / 1024.0) as f32 + 1e-6).sqrt();
        for channel in 0..1024 {
            output[channel] *= scale;
            output[channel] *= block.norm_w[channel];
            output[channel] += block.norm_b[channel];
        }
    }
    let expanded = matmul_2d_pw(
        &block.pw1_w,
        Some(&block.pw1_b),
        &normalized,
        4096,
        1024,
        upsampled_len,
    )?;
    let expanded = gelu_inplace(expanded);
    let projected = matmul_2d_pw(
        &block.pw2_w,
        Some(&block.pw2_b),
        &expanded,
        1024,
        4096,
        upsampled_len,
    )?;
    let mut out = upsampled.clone();
    for (row, projected_row) in out.chunks_exact_mut(1024).zip(projected.chunks_exact(1024)) {
        for channel in 0..1024 {
            row[channel] += block.gamma[channel] * projected_row[channel];
        }
    }
    Ok(out)
}

fn matmul_2d_pw(
    weight: &[u8],
    bias: Option<&[f32]>,
    input: &[f32],
    out_dim: usize,
    in_dim: usize,
    length: usize,
) -> Result<Vec<f32>, String> {
    let blocks = in_dim / 32;
    let expected_weight = blocks * out_dim * 34;
    if in_dim % 32 != 0 || weight.len() != expected_weight {
        return Err(format!(
            "matmul_2d_pw: weight len {} != Q8_0 matrix {}x{}",
            weight.len(),
            in_dim,
            out_dim,
        ));
    }
    if input.len() != in_dim * length {
        return Err("matmul_2d_pw: input length mismatch".into());
    }
    if bias.is_some_and(|values| values.len() != out_dim) {
        return Err("matmul_2d_pw: bias length mismatch".into());
    }
    let mut out = vec![0.0f32; length * out_dim];
    for t in 0..length {
        let mut input_q8 = vec![0u8; in_dim];
        let mut input_scales = vec![0.0f32; blocks];
        quantize_q8_0_into(
            &input[t * in_dim..(t + 1) * in_dim],
            in_dim,
            &mut input_q8,
            &mut input_scales,
        );
        let row = &mut out[t * out_dim..(t + 1) * out_dim];
        matmul_q8_0_quantized_parallel(weight, &input_q8, &input_scales, row, in_dim, out_dim);
        if let Some(bias) = bias {
            for (value, bias) in row.iter_mut().zip(bias) {
                *value += *bias;
            }
        }
    }
    Ok(out)
}

fn gelu_inplace(mut x: Vec<f32>) -> Vec<f32> {
    for v in x.iter_mut() {
        if *v <= -10.0 {
            *v = 0.0;
        } else if *v < 10.0 {
            let value = f16_to_f32(f32_to_f16(*v));
            let inner = 0.7978845608028654 * value * (1.0 + 0.044715 * value * value);
            #[cfg(unix)]
            let activation = unsafe { tanhf(inner) };
            #[cfg(not(unix))]
            let activation = inner.tanh();
            *v = f16_to_f32(f32_to_f16(0.5 * value * (1.0 + activation)));
        }
    }
    x
}

fn dac_block_channels(block_idx: usize) -> Result<(usize, usize, usize), String> {
    let channels: [(usize, usize, usize); 4] = [
        (1536, 768, DAC_BLOCK_KERNELS[0]),
        (768, 384, DAC_BLOCK_KERNELS[1]),
        (384, 192, DAC_BLOCK_KERNELS[2]),
        (192, 96, DAC_BLOCK_KERNELS[3]),
    ];
    channels
        .get(block_idx)
        .copied()
        .ok_or_else(|| format!("unknown DAC block index {block_idx}"))
}

fn dac_residual_conv1_kernel(_block_idx: usize, _res_idx: usize) -> usize {
    // All four DAC blocks use kernel=7 for the conv1 in each residual unit
    // (per dump_tensors output); dilations vary via `DAC_DILATIONS`.
    7
}

fn load_f32_tensor_3d(
    source: &dyn TensorSource,
    name: &str,
    expected_k: usize,
    expected_in: usize,
    expected_out: usize,
) -> Result<Vec<f32>, String> {
    let bytes = source
        .tensor_slice(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing tensor info: {name}"))?;
    let dims = [expected_k as u64, expected_in as u64, expected_out as u64];
    if info.dims != dims {
        return Err(format!("{name}: dims {:?} != expected {dims:?}", info.dims,));
    }
    if info.ggml_type != GGMLType::F32 {
        return Err(format!("{name}: type {:?} not F32", info.ggml_type));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}
