//! 4-stage DAC upsampler (Descript Audio Codec) plus 2 ConvNeXt upsample
//! blocks for the Qwen3-TTS codec.
//!
//! Full decode pipeline (C-first layout, `[channels, length]`):
//!
//! 1. `pre_conv` (Conv1d k=3): 512 → 1024 channels.
//! 2. **2 upsample stages** (each): causal ConvTranspose1d (kernel=2, stride=2)
//!    followed by a ConvNeXt block (dwconv k=7 → LayerNorm → pwconv1 4× → GELU
//!    → pwconv2 → per-channel gamma).
//! 3. `dac_entry`: Conv1d k=7, 1024 → 1536.
//! 4. **4 DAC blocks**: Snake → causal ConvTranspose1d (kernel=2×stride, stride
//!    in {8, 5, 4, 3}) → 3 residual units with dilations `[1, 3, 9]`.
//! 5. `dac_post`: Snake → Conv1d k=7, 96 → 1 channel.
//! 6. Clamp to `[-1, 1]`.

use crate::core::tensor::{GGMLType, TensorSource};
use crate::models::qwen3::{
    checked_product, load_f32_tensor, usize_to_u64,
};
use crate::models::tts::codec::conv::{conv1d, conv_transpose1d};
use crate::models::tts::codec::snake::snake1d_inplace;
use crate::models::tts::load_f16_or_f32_tensor;

const DAC_ENTRY_KERNEL: usize = 7;
const DAC_POST_KERNEL: usize = 7;
const DAC_BLOCK_KERNELS: [usize; 4] = [16, 10, 8, 6];
const DAC_BLOCK_STRIDES: [usize; 4] = [8, 5, 4, 3];
const DAC_DILATIONS: [usize; 3] = [1, 3, 9];
const UPSAMPLE_KERNEL: usize = 2;
const UPSAMPLE_STRIDE: usize = 2;
const CONVNEXT_KERNEL: usize = 7;
const CONVNEXT_EXPANSION: usize = 4;

pub struct DacDecoder {
    /// Conv1d 512 → 1024, kernel 3 (length shrinks by 2).
    pre_conv_w: Vec<f32>,
    pre_conv_b: Vec<f32>,
    /// 2 ConvNeXt upsample stages (each 2× upsampling).
    upsample_blocks: Vec<UpsampleBlock>,
    /// Conv1d 1024 → 1536, kernel 7.
    entry_weight: Vec<f32>,
    entry_bias: Vec<f32>,
    /// 4 Snake + ConvTranspose1d + 3-residual-unit blocks.
    blocks: Vec<DacBlock>,
    /// Snake + Conv1d 96 → 1, kernel 7.
    post_snake_alpha: Vec<f32>,
    post_snake_beta: Vec<f32>,
    post_weight: Vec<f32>,
    post_bias: Vec<f32>,
}

pub struct UpsampleBlock {
    /// Causal ConvTranspose1d k=2, stride=2 (channel-preserving).
    conv_w: Vec<f32>,
    conv_b: Vec<f32>,
    /// Depthwise Conv1d k=7 (per-channel, single group).
    dwconv_w: Vec<f32>,
    dwconv_b: Vec<f32>,
    /// LayerNorm over channel dim.
    norm_w: Vec<f32>,
    norm_b: Vec<f32>,
    /// Pointwise Conv1d: pw1 (1024 → 4096), pw2 (4096 → 1024).
    pw1_w: Vec<f32>,
    pw1_b: Vec<f32>,
    pw2_w: Vec<f32>,
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
    conv1_weight: Vec<f32>,
    conv1_bias: Vec<f32>,
    dilation: usize,
    act2_alpha: Vec<f32>,
    act2_beta: Vec<f32>,
    conv2_weight: Vec<f32>,
    conv2_bias: Vec<f32>,
}

impl DacDecoder {
    pub fn from_source(source: &dyn TensorSource) -> Result<Self, String> {
        Self::from_source_dyn(source)
    }
}

impl DacDecoder {
    fn from_source_dyn(source: &dyn TensorSource) -> Result<Self, String> {
        let pre_conv_w = load_f16_or_f32_tensor(
            source,
            "a.gen.wav.pre_conv.weight",
            &[3, 512, 1024],
        )?;
        let pre_conv_b = load_f32_tensor(
            source,
            "a.gen.wav.pre_conv.bias",
            &[usize_to_u64(1024, "pre_conv bias")?],
        )?;
        // The reference decoder expects 1024-dim input to the DAC entry conv,
        // so the DAC's `decode` method starts at 1024-dim. The 512 → 1024
        // pre_conv is exposed separately (via [`Self::pre_conv`]) for callers
        // that have 512-dim RVQ output (the common case).
        let entry_weight = load_f16_or_f32_tensor(
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

    /// Apply the 512 → 1024 pre_conv (causal Conv1d k=3) and return the new
    /// `(length, 1024)` C-first buffer. Length is preserved (zero-pad on the
    /// left) so the waveform TFM sees the same temporal extent as the input.
    pub fn pre_conv(&self, input: &[f32], length: usize) -> Result<Vec<f32>, String> {
        if input.len() != 512 * length {
            return Err(format!(
                "DacDecoder.pre_conv: input length {} != 512 * length {}",
                input.len(),
                length,
            ));
        }
        let out = conv1d_causal_rearranged(
            &self.pre_conv_w,
            Some(&self.pre_conv_b),
            input,
            512,
            length,
            1024,
            3,
        )?;
        Ok(out)
    }

    /// Decode `[1024, length]` waveform embedding into a 1-channel PCM buffer
    /// (after clamp to `[-1, 1]`). Caller should pass the output of
    /// [`Self::pre_conv`] followed by the waveform transformer's `out_proj`.
    pub fn decode(&self, input: &[f32], length: usize) -> Result<Vec<f32>, String> {
        if input.len() != 1024 * length {
            return Err(format!(
                "DacDecoder: input length {} != 1024 * length {}",
                input.len(),
                length,
            ));
        }

        // 1. Two ConvNeXt upsample stages (each 2×).
        let mut current = input.to_vec();
        let mut current_len = length;
        for (block_idx, block) in self.upsample_blocks.iter().enumerate() {
            current = upsample_block_forward(
                block,
                &current,
                current_len,
                block_idx,
            )?;
            current_len = current.len() / 1024;
        }

        // 2. DAC entry: Conv1d 1024 -> 1536 with kernel 7.
        let (mut cur_len, mut cur) = conv1d_rearranged(
            &self.entry_weight,
            Some(&self.entry_bias),
            &current,
            1024,
            current_len,
            1536,
            DAC_ENTRY_KERNEL,
        )?;

        // 4. 4 DAC blocks.
        for block in &self.blocks {
            cur = block.forward(&cur, cur_len)?;
            cur_len = cur.len() / block.out_channels;
        }
        let block_last_out = self.blocks.last().unwrap().out_channels;
        let block_last_len = cur.len() / block_last_out;

        // 5. Post: Snake + Conv1d 96 -> 1.
        snake1d_inplace(&mut cur, block_last_len, &self.post_snake_alpha, &self.post_snake_beta)?;
        let (_post_len, post_out) = conv1d_rearranged(
            &self.post_weight,
            Some(&self.post_bias),
            &cur,
            block_last_out,
            block_last_len,
            1,
            DAC_POST_KERNEL,
        )?;
        // post_out is [1, post_len]. The reference decoder then clamps to
        // [-1, 1]; we leave clamping to the WAV writer (which clamps to i16).
        let mut audio = post_out;
        for v in audio.iter_mut() {
            *v = v.clamp(-1.0, 1.0);
        }
        Ok(audio)
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
        // Snake first (operates on in_channels because Snake input dim = pre-ConvT channel count).
        let mut after_snake = input.to_vec();
        snake1d_inplace(
            &mut after_snake,
            length,
            &self.snake_alpha,
            &self.snake_beta,
        )?;
        // ConvTranspose1d with kernel=2*stride; the (K-stride) tail is
        // discarded because at startup the prior tail is empty. We use a
        // simpler full-padding implementation here for correctness.
        let (upsampled_len, mut upsampled) = conv_transpose1d(
            &self.conv_weight,
            Some(&self.conv_bias),
            &after_snake,
            self.in_channels,
            length,
            self.out_channels,
            self.kernel_size,
            self.stride,
        )?;
        // Apply residual units with dilations [1, 3, 9].
        let mut cur = upsampled;
        let mut cur_len = upsampled_len;
        for (i, residual) in self.residual.iter().enumerate() {
            cur = residual.forward(&cur, cur_len, DAC_DILATIONS[i])?;
            // Residual units preserve length.
        }
        Ok(cur)
    }
}

impl DacResidual {
    fn forward(
        &self,
        input: &[f32],
        length: usize,
        dilation: usize,
    ) -> Result<Vec<f32>, String> {
        let ch = self.act1_alpha.len();
        if input.len() != ch * length {
            return Err(format!(
                "DacResidual.forward: input length {} != expected {}",
                input.len(),
                ch * length,
            ));
        }
        // Snake act1
        let mut after_act1 = input.to_vec();
        snake1d_inplace(&mut after_act1, length, &self.act1_alpha, &self.act1_beta)?;
        // Dilated Conv1d ch -> ch, kernel 7 (or 3 for blk.3).
        let kernel_size = self.conv1_weight.len() / (ch * ch);
        let (after_conv1_len, after_conv1) = conv1d_dilated_rearranged(
            &self.conv1_weight,
            Some(&self.conv1_bias),
            &after_act1,
            ch,
            length,
            ch,
            kernel_size,
            dilation,
        )?;
        // Snake act2
        let mut after_act2 = after_conv1;
        snake1d_inplace(&mut after_act2, after_conv1_len, &self.act2_alpha, &self.act2_beta)?;
        // Conv1d ch -> ch kernel 1.
        let (after_conv2_len, after_conv2) = conv1d_rearranged(
            &self.conv2_weight,
            Some(&self.conv2_bias),
            &after_act2,
            ch,
            after_conv1_len,
            ch,
            1,
        )?;
        // Residual add (lengths match because the conv kernel of conv2 is 1).
        debug_assert_eq!(after_conv2_len, length);
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

fn load_upsample_block(source: &dyn TensorSource, block_idx: usize) -> Result<UpsampleBlock, String> {
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
    let dwconv_w = load_f16_or_f32_tensor(
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
    let pw1_w = load_q8_2d(
        source,
        &format!("{prefix}.pw1.weight"),
        1024,
        4096,
    )?;
    let pw1_b = load_f32_tensor(
        source,
        &format!("{prefix}.pw1.bias"),
        &[usize_to_u64(4096, "upsample pw1 bias")?],
    )?;
    let pw2_w = load_q8_2d(
        source,
        &format!("{prefix}.pw2.weight"),
        4096,
        1024,
    )?;
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
    block_idx: usize,
) -> Result<Vec<f32>, String> {
    // ConvTranspose1d (k=2, stride=2, channel-preserving 1024->1024).
    let (upsampled_len, mut upsampled) = conv_transpose1d(
        &block.conv_w,
        Some(&block.conv_b),
        input,
        1024,
        length,
        1024,
        UPSAMPLE_KERNEL,
        UPSAMPLE_STRIDE,
    )?;
    // ConvNeXt block: depthwise conv1d k=7 → LayerNorm → pw1 → GELU → pw2 → gamma → residual.
    // The depthwise conv uses one filter per output channel (groups=in_channels),
    // so the kernel has shape [K, 1, C] in GGUF = K * C values total.
    let (dwconv_len, dwconv_out) = depthwise_conv1d_causal_rearranged(
        &block.dwconv_w,
        Some(&block.dwconv_b),
        &upsampled,
        1024,
        upsampled_len,
        CONVNEXT_KERNEL,
    )?;
    // Apply per-channel LayerNorm to dwconv output (note: weights layout is [C]
    // in this codebase's convention).
    let mut normalized = vec![0.0f32; dwconv_out.len()];
    for c in 0..1024 {
        let row = &dwconv_out[c * dwconv_len..(c + 1) * dwconv_len];
        let mean: f32 = row.iter().sum::<f32>() / dwconv_len as f32;
        let var: f32 =
            row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / dwconv_len as f32;
        let std = (var + 1e-6).sqrt();
        let out_row = &mut normalized[c * dwconv_len..(c + 1) * dwconv_len];
        for t in 0..dwconv_len {
            let v = (row[t] - mean) / std;
            out_row[t] = v * block.norm_w[c] + block.norm_b[c];
        }
    }
    // pw1 (1024 -> 4096) and pw2 (4096 -> 1024) are matmuls over the channel dim.
    // Apply per-token matmul (no length mixing): out[o, t] = sum_i w[o, i] * in[i, t].
    let expanded = matmul_2d_pw(&block.pw1_w, &normalized, 4096, 1024, dwconv_len)?;
    let expanded = gelu_inplace(expanded);
    let projected = matmul_2d_pw(&block.pw2_w, &expanded, 1024, 4096, dwconv_len)?;
    // Apply per-channel gamma and add residual.
    let mut out = upsampled.clone();
    for c in 0..1024 {
        let row_in = &projected[c * dwconv_len..(c + 1) * dwconv_len];
        let row_out = &mut out[c * dwconv_len..(c + 1) * dwconv_len];
        for t in 0..dwconv_len {
            row_out[t] += block.gamma[c] * row_in[t];
        }
    }
    let _ = block_idx; // unused, kept for symmetry with reference.
    Ok(out)
}

fn matmul_2d_pw(
    weight: &[f32],
    input: &[f32],
    out_dim: usize,
    in_dim: usize,
    length: usize,
) -> Result<Vec<f32>, String> {
    if weight.len() != out_dim * in_dim {
        return Err(format!(
            "matmul_2d_pw: weight len {} != {}*{}",
            weight.len(),
            out_dim,
            in_dim
        ));
    }
    if input.len() != in_dim * length {
        return Err("matmul_2d_pw: input length mismatch".into());
    }
    let mut out = vec![0.0f32; out_dim * length];
    for t in 0..length {
        for o in 0..out_dim {
            let mut acc = 0.0f32;
            for i in 0..in_dim {
                acc += weight[o * in_dim + i] * input[i * length + t];
            }
            out[o * length + t] = acc;
        }
    }
    Ok(out)
}

fn gelu_inplace(mut x: Vec<f32>) -> Vec<f32> {
    for v in x.iter_mut() {
        let val = *v;
        *v = 0.5 * val
            * (1.0 + (val * 0.7978845608028654 * (1.0 + 0.044715 * val * val)).tanh());
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

/// Conv1d with dilation (zero padding on the left so that the right edge is
/// preserved — used by the DAC residual units with dilations `[1, 3, 9]`).
fn conv1d_dilated_rearranged(
    kernel: &[f32],
    bias: Option<&[f32]>,
    input: &[f32],
    in_channels: usize,
    length_in: usize,
    out_channels: usize,
    kernel_size: usize,
    dilation: usize,
) -> Result<(usize, Vec<f32>), String> {
    let length_out = length_in;
    let expected_kernel = in_channels * out_channels * kernel_size;
    if kernel.len() != expected_kernel {
        return Err(format!(
            "conv1d_dilated_rearranged: kernel len {} != expected {}",
            kernel.len(),
            expected_kernel,
        ));
    }
    let mut rearranged = vec![0.0f32; expected_kernel];
    for k in 0..kernel_size {
        for ic in 0..in_channels {
            for oc in 0..out_channels {
                let src = k * in_channels * out_channels + ic * out_channels + oc;
                let dst = oc * in_channels * kernel_size + ic * kernel_size + k;
                rearranged[dst] = kernel[src];
            }
        }
    }
    let pad = (kernel_size - 1) * dilation;
    let mut padded = vec![0.0f32; in_channels * (length_in + pad)];
    for ic in 0..in_channels {
        let src_start = ic * length_in;
        let dst_start = ic * (length_in + pad) + pad;
        for t in 0..length_in {
            padded[dst_start + t] = input[src_start + t];
        }
    }
    let mut output = vec![0.0f32; out_channels * length_out];
    for oc in 0..out_channels {
        let bias_val = bias.map_or(0.0, |b| b[oc]);
        let base = oc * length_out;
        for o in 0..length_out {
            let mut acc = bias_val;
            for ic in 0..in_channels {
                for k in 0..kernel_size {
                    let kernel_idx = oc * in_channels * kernel_size + ic * kernel_size + k;
                    let input_idx = ic * (length_in + pad) + o + k * dilation;
                    acc += rearranged[kernel_idx] * padded[input_idx];
                }
            }
            output[base + o] = acc;
        }
    }
    Ok((length_out, output))
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

/// Causal depthwise Conv1d: each output channel uses its own filter
/// (`groups = in_channels`). The kernel has shape `[K, 1, C]` in GGUF (=
/// `K * C` f32 values total), and we left-pad with `K - 1` zeros so the
/// output length matches the input length.
fn depthwise_conv1d_causal_rearranged(
    kernel: &[f32],
    bias: Option<&[f32]>,
    input: &[f32],
    channels: usize,
    length_in: usize,
    kernel_size: usize,
) -> Result<(usize, Vec<f32>), String> {
    let expected_kernel = kernel_size * channels;
    if kernel.len() != expected_kernel {
        return Err(format!(
            "depthwise_conv1d_causal_rearranged: kernel len {} != expected {}",
            kernel.len(),
            expected_kernel
        ));
    }
    if input.len() != channels * length_in {
        return Err("depthwise_conv1d_causal_rearranged: input length mismatch".into());
    }
    // GGUF stores depthwise as [K, 1, C]; rearrange to per-channel weight
    // vectors [C, K] for easier indexing below.
    let mut w = vec![0.0f32; expected_kernel];
    for k in 0..kernel_size {
        for c in 0..channels {
            let src = k * channels + c;
            let dst = c * kernel_size + k;
            w[dst] = kernel[src];
        }
    }
    let pad = kernel_size - 1;
    let padded_len = length_in + pad;
    let mut padded = vec![0.0f32; channels * padded_len];
    for c in 0..channels {
        let src_start = c * length_in;
        let dst_start = c * padded_len + pad;
        padded[dst_start..dst_start + length_in]
            .copy_from_slice(&input[src_start..src_start + length_in]);
    }
    let mut output = vec![0.0f32; channels * length_in];
    for c in 0..channels {
        let bias_val = bias.map_or(0.0, |b| b[c]);
        let row_w = &w[c * kernel_size..(c + 1) * kernel_size];
        let base = c * length_in;
        for t in 0..length_in {
            let mut acc = bias_val;
            for k in 0..kernel_size {
                acc += row_w[k] * padded[c * padded_len + t + k];
            }
            output[base + t] = acc;
        }
    }
    Ok((length_in, output))
}

/// Causal Conv1d: pads the input on the LEFT with `kernel_size - 1` zeros so
/// that the output has the same length as the input. The kernel is stored in
/// the GGUF layout `[k, in, out]` (reverse of PyTorch's `[out, in, k]`).
fn conv1d_causal_rearranged(
    kernel: &[f32],
    bias: Option<&[f32]>,
    input: &[f32],
    in_channels: usize,
    length_in: usize,
    out_channels: usize,
    kernel_size: usize,
) -> Result<Vec<f32>, String> {
    let expected_kernel = in_channels * out_channels * kernel_size;
    if kernel.len() != expected_kernel {
        return Err(format!(
            "conv1d_causal_rearranged: kernel len {} != expected {}",
            kernel.len(),
            expected_kernel,
        ));
    }
    if input.len() != in_channels * length_in {
        return Err("conv1d_causal_rearranged: input length mismatch".into());
    }
    // Rearrange GGUF `[k, in, out]` → PyTorch Conv1d `[out, in, k]`.
    let mut rearranged = vec![0.0f32; expected_kernel];
    for k in 0..kernel_size {
        for ic in 0..in_channels {
            for oc in 0..out_channels {
                let src_idx = k * in_channels * out_channels + ic * out_channels + oc;
                let dst_idx = oc * in_channels * kernel_size + ic * kernel_size + k;
                rearranged[dst_idx] = kernel[src_idx];
            }
        }
    }
    let pad = kernel_size - 1;
    // Left-pad the input: zeros on the left of each channel's row.
    let padded_len = length_in + pad;
    let mut padded = vec![0.0f32; in_channels * padded_len];
    for ic in 0..in_channels {
        let src_start = ic * length_in;
        let dst_start = ic * padded_len + pad;
        padded[dst_start..dst_start + length_in]
            .copy_from_slice(&input[src_start..src_start + length_in]);
    }
    let mut output = vec![0.0f32; out_channels * length_in];
    for oc in 0..out_channels {
        let bias_val = bias.map_or(0.0, |b| b[oc]);
        let base = oc * length_in;
        for o in 0..length_in {
            let mut acc = bias_val;
            for ic in 0..in_channels {
                for k in 0..kernel_size {
                    let kernel_idx = oc * in_channels * kernel_size + ic * kernel_size + k;
                    let input_idx = ic * padded_len + o + k;
                    acc += rearranged[kernel_idx] * padded[input_idx];
                }
            }
            output[base + o] = acc;
        }
    }
    Ok(output)
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
        return Err(format!(
            "{name}: dims {:?} != expected {dims:?}",
            info.dims,
        ));
    }
    if info.ggml_type != GGMLType::F32 {
        return Err(format!("{name}: type {:?} not F32", info.ggml_type));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn load_q8_2d(
    source: &dyn TensorSource,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<Vec<f32>, String> {
    let bytes = source
        .tensor_slice(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing tensor info: {name}"))?;
    let dims = [rows as u64, cols as u64];
    if info.dims != dims {
        return Err(format!("{name}: dims {:?} != expected {dims:?}", info.dims));
    }
    if info.ggml_type != GGMLType::Q8_0 {
        return Err(format!("{name}: type {:?} not Q8_0", info.ggml_type));
    }
    let blocks_per_row = cols / 32;
    let bytes_per_row = blocks_per_row * 34;
    let expected = checked_product("q8 bytes", rows, bytes_per_row)?;
    if bytes.len() != expected {
        return Err(format!("{name}: bytes {} != expected {expected}", bytes.len()));
    }
    let mut out = vec![0.0f32; rows * cols];
    for row in 0..rows {
        for b in 0..blocks_per_row {
            let off = row * bytes_per_row + b * 34;
            let scale = half::f16::from_le_bytes([bytes[off], bytes[off + 1]]).to_f32();
            for j in 0..32usize {
                let q = bytes[off + 2 + j] as i8 as f32;
                out[row * cols + b * 32 + j] = scale * q;
            }
        }
    }
    Ok(out)
}
