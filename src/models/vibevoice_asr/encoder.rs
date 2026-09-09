//! VibeVoice speech tokenizers (ConvNeXt-style causal ConvNet encoders) and
//! speech connectors, mirroring the official `TokenizerEncoder` /
//! `SpeechConnector` from microsoft/VibeVoice.
//!
//! All activations are token-major (`[T][C]` row-major) so the per-frame
//! channel RMSNorm, depthwise mixer and FFN matmuls work on contiguous rows.
//! Weighted convolutions are evaluated as im2col + native row-major matmul;
//! the per-channel mixer is a direct kernel-tap FIR. The causal `SConv1d`
//! padding follows the official `padding_total = kernel - stride` (left) plus
//! the stride-alignment `extra` right padding.

use crate::core::tensor::{load_f32_tensor, GGMLType, TensorSource};
use crate::ops::{rms_norm, rms_norm_inplace};

unsafe extern "C" {
    fn erff(value: f32) -> f32;
}

/// transformers `ACT2FN["gelu"]` — the exact erf form.
fn gelu_erf_inplace(values: &mut [f32]) {
    for value in values.iter_mut() {
        *value = 0.5 * *value * (1.0 + unsafe { erff(*value * std::f32::consts::FRAC_1_SQRT_2) });
    }
}

// --------------------------------------------------------------------------- //
// GGUF tensor loading helpers (BF16 or F32 storage → f32)
// --------------------------------------------------------------------------- //

fn load_tensor_f32(
    source: &dyn TensorSource,
    name: &str,
    dims: &[u64],
) -> Result<Vec<f32>, String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    if info.dims != dims {
        return Err(format!(
            "Invalid tensor {name}: dims {:?}; expected {:?}",
            info.dims, dims
        ));
    }
    let bytes = source
        .tensor_slice(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    let elements = dims.iter().product::<u64>() as usize;
    match info.ggml_type {
        GGMLType::BF16 => {
            if bytes.len() != elements * 2 {
                return Err(format!("Invalid tensor data length for {name}"));
            }
            Ok(bytes
                .chunks_exact(2)
                .map(|chunk| {
                    crate::core::tensor::bf16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]))
                })
                .collect())
        }
        GGMLType::F32 => {
            if bytes.len() != elements * 4 {
                return Err(format!("Invalid tensor data length for {name}"));
            }
            Ok(bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect())
        }
        other => Err(format!(
            "Invalid tensor {name}: type {other:?}; expected BF16 or F32"
        )),
    }
}

fn expect_f32_vector(source: &dyn TensorSource, name: &str, len: u64) -> Result<Vec<f32>, String> {
    load_f32_tensor(source, name, &[len])
}

// --------------------------------------------------------------------------- //
// dense (row-major [n_out][n_in]) matmul
// --------------------------------------------------------------------------- //

pub struct Dense {
    pub data: Vec<f32>,
    pub n_in: usize,
    pub n_out: usize,
}

impl Dense {
    fn from_source(
        source: &dyn TensorSource,
        name: &str,
        n_in: usize,
        n_out: usize,
    ) -> Result<Self, String> {
        Ok(Self {
            data: load_tensor_f32(
                source,
                &format!("{name}.weight"),
                &[n_in as u64, n_out as u64],
            )?,
            n_in,
            n_out,
        })
    }

    /// output[rows, n_out] = input[rows, n_in] · Wᵀ + bias
    fn forward_rows(&self, input: &[f32], rows: usize, bias: &[f32], output: &mut [f32]) {
        debug_assert_eq!(input.len(), rows * self.n_in);
        debug_assert_eq!(output.len(), rows * self.n_out);
        debug_assert_eq!(bias.len(), self.n_out);
        for row in output.chunks_exact_mut(self.n_out) {
            row.copy_from_slice(bias);
        }
        for (in_row, out_row) in input
            .chunks_exact(self.n_in)
            .zip(output.chunks_exact_mut(self.n_out))
        {
            for (out_value, weight_row) in out_row.iter_mut().zip(self.data.chunks_exact(self.n_in))
            {
                let mut sum = 0.0f32;
                for (&input_value, &weight_value) in in_row.iter().zip(weight_row) {
                    sum = weight_value.mul_add(input_value, sum);
                }
                *out_value += sum;
            }
        }
    }
}

// --------------------------------------------------------------------------- //
// causal convolution weights ([n_out][n_in][kernel], flattened rows)
// --------------------------------------------------------------------------- //

pub struct Conv1d {
    /// Flattened torch conv weight: one row per output channel, [c_in][kernel].
    pub data: Vec<f32>,
    pub n_in: usize,
    pub n_out: usize,
    pub kernel: usize,
    pub bias: Vec<f32>,
}

impl Conv1d {
    fn from_source(
        source: &dyn TensorSource,
        name: &str,
        n_in: usize,
        n_out: usize,
        kernel: usize,
    ) -> Result<Self, String> {
        Ok(Self {
            data: load_tensor_f32(
                source,
                &format!("{name}.weight"),
                &[kernel as u64, n_in as u64, n_out as u64],
            )?,
            bias: expect_f32_vector(source, &format!("{name}.bias"), n_out as u64)?,
            n_in,
            n_out,
            kernel,
        })
    }

    /// Causal non-streaming forward over token-major `x` ([t_in][n_in]).
    ///
    /// Mirrors the official `SConv1d._forward_non_streaming` for a causal,
    /// constant-padded, dilation-1 conv: `padding_total = kernel - stride` on
    /// the left plus the stride-alignment `extra` padding on the right.
    fn forward_token_major(&self, x: &[f32], t_in: usize, stride: usize, output: &mut Vec<f32>) {
        let k = self.kernel;
        let padding_total = k - stride;
        let n_frames = (t_in as f64 - k as f64 + padding_total as f64) / stride as f64 + 1.0;
        let ideal = (n_frames.ceil() as usize - 1) * stride + (k - padding_total);
        let extra = ideal.saturating_sub(t_in);
        let t_out = 1 + (padding_total + t_in + extra - k) / stride;

        output.clear();
        output.resize(t_out * self.n_out, 0.0);
        for row in output.chunks_exact_mut(self.n_out) {
            row.copy_from_slice(&self.bias);
        }

        let row_width = self.n_in * k;
        let mut patches = vec![0.0f32; t_out * row_width];
        for t in 0..t_out {
            let patch_base = t * row_width;
            for kk in 0..k {
                let padded = t * stride + kk;
                if padded < padding_total || padded >= padding_total + t_in {
                    continue; // zero padding contributes nothing
                }
                let source_index = padded - padding_total;
                let row = &x[source_index * self.n_in..(source_index + 1) * self.n_in];
                let patch = &mut patches[patch_base..patch_base + row_width];
                for (ci, &value) in row.iter().enumerate() {
                    patch[ci * k + kk] = value;
                }
            }
        }
        self.gemm_rows(&patches, t_out, output);
    }

    /// output += patches · Wᵀ (accumulates over the bias-seeded output).
    fn gemm_rows(&self, patches: &[f32], t_out: usize, output: &mut [f32]) {
        let row_width = self.n_in * self.kernel;
        for (patch_row, out_row) in patches
            .chunks_exact(row_width)
            .zip(output.chunks_exact_mut(self.n_out))
        {
            for (out_value, weight_row) in out_row.iter_mut().zip(self.data.chunks_exact(row_width))
            {
                let mut sum = 0.0f32;
                for (&patch_value, &weight_value) in patch_row.iter().zip(weight_row) {
                    sum = weight_value.mul_add(patch_value, sum);
                }
                *out_value += sum;
            }
        }
    }
}

// --------------------------------------------------------------------------- //
// ConvNeXt block
// --------------------------------------------------------------------------- //

pub struct EncoderBlock {
    norm: Vec<f32>,
    mixer: Conv1d, // depthwise: [dim][1][kernel]
    gamma: Vec<f32>,
    ffn_norm: Vec<f32>,
    ffn_in: Dense,
    ffn_in_bias: Vec<f32>,
    ffn_out: Dense,
    ffn_out_bias: Vec<f32>,
    ffn_gamma: Vec<f32>,
}

pub struct BlockScratch {
    normed: Vec<f32>,
    conv_out: Vec<f32>,
    hidden: Vec<f32>,
    buffer: Vec<f32>,
}

impl BlockScratch {
    fn new(t: usize, dim: usize, ffn_dim: usize) -> Self {
        Self {
            normed: vec![0.0; t * dim],
            conv_out: vec![0.0; t * dim],
            hidden: vec![0.0; t * ffn_dim],
            buffer: vec![0.0; t * dim],
        }
    }
}

impl EncoderBlock {
    #[allow(clippy::too_many_arguments)]
    fn from_source(
        source: &dyn TensorSource,
        name: &str,
        dim: usize,
        ffn_dim: usize,
        kernel: usize,
    ) -> Result<Self, String> {
        Ok(Self {
            norm: expect_f32_vector(source, &format!("{name}.norm.weight"), dim as u64)?,
            mixer: Conv1d::from_source(source, &format!("{name}.mixer"), 1, dim, kernel)?,
            gamma: expect_f32_vector(source, &format!("{name}.gamma"), dim as u64)?,
            ffn_norm: expect_f32_vector(source, &format!("{name}.ffn_norm.weight"), dim as u64)?,
            ffn_in: Dense::from_source(source, &format!("{name}.ffn_linear1"), dim, ffn_dim)?,
            ffn_in_bias: expect_f32_vector(
                source,
                &format!("{name}.ffn_linear1.bias"),
                ffn_dim as u64,
            )?,
            ffn_out: Dense::from_source(source, &format!("{name}.ffn_linear2"), ffn_dim, dim)?,
            ffn_out_bias: expect_f32_vector(
                source,
                &format!("{name}.ffn_linear2.bias"),
                dim as u64,
            )?,
            ffn_gamma: expect_f32_vector(source, &format!("{name}.ffn_gamma"), dim as u64)?,
        })
    }

    /// In-place block update over token-major `x` ([t][dim]).
    fn forward(&self, x: &mut [f32], t: usize, dim: usize, scratch: &mut BlockScratch) {
        // mixer branch: RMSNorm over channels → depthwise causal conv → scale
        for (token, normed_row) in scratch.normed[..t * dim].chunks_exact_mut(dim).enumerate() {
            rms_norm(
                &x[token * dim..(token + 1) * dim],
                &self.norm,
                normed_row,
                LN_EPS,
            );
        }
        depthwise_conv_causal(
            &scratch.normed[..t * dim],
            &self.mixer.data,
            &self.mixer.bias,
            t,
            dim,
            self.mixer.kernel,
            &mut scratch.conv_out,
        );
        for (token, out_row) in scratch.conv_out[..t * dim].chunks_exact(dim).enumerate() {
            for (channel, &value) in out_row.iter().enumerate() {
                x[token * dim + channel] += value * self.gamma[channel];
            }
        }

        // FFN branch: RMSNorm → fc1 → gelu(erf) → fc2 → scale
        for (token, normed_row) in scratch.normed[..t * dim].chunks_exact_mut(dim).enumerate() {
            rms_norm(
                &x[token * dim..(token + 1) * dim],
                &self.ffn_norm,
                normed_row,
                LN_EPS,
            );
        }
        self.ffn_in.forward_rows(
            &scratch.normed[..t * dim],
            t,
            &self.ffn_in_bias,
            &mut scratch.hidden,
        );
        gelu_erf_inplace(&mut scratch.hidden[..t * self.ffn_in.n_out]);
        self.ffn_out
            .forward_rows(&scratch.hidden, t, &self.ffn_out_bias, &mut scratch.buffer);
        for (token, out_row) in scratch.buffer[..t * dim].chunks_exact(dim).enumerate() {
            for (channel, &value) in out_row.iter().enumerate() {
                x[token * dim + channel] += value * self.ffn_gamma[channel];
            }
        }
    }
}

/// Depthwise causal conv (stride 1, constant zero left padding) over
/// token-major data. `weight` rows are per-channel kernel taps.
fn depthwise_conv_causal(
    x: &[f32],
    weight: &[f32],
    bias: &[f32],
    t: usize,
    dim: usize,
    kernel: usize,
    output: &mut Vec<f32>,
) {
    output.clear();
    output.resize(t * dim, 0.0);
    for token in 0..t {
        let out_row = &mut output[token * dim..(token + 1) * dim];
        out_row.copy_from_slice(bias);
        for tap in 0..kernel {
            let source = token + tap;
            if source < kernel - 1 {
                continue; // zero left padding
            }
            let source = source - (kernel - 1);
            if source >= t {
                continue;
            }
            let input_row = &x[source * dim..(source + 1) * dim];
            for (channel, &input_value) in input_row.iter().enumerate() {
                out_row[channel] += weight[channel * kernel + tap] * input_value;
            }
        }
    }
}

// --------------------------------------------------------------------------- //
// TokenizerEncoder
// --------------------------------------------------------------------------- //

/// Block ConvRMSNorm epsilon (the checkpoint's `layernorm_eps`).
const LN_EPS: f32 = 1e-5;

pub struct TokenizerEncoder {
    downsample: Vec<Conv1d>,
    /// Runtime strides: 1 for the stem, then the reversed checkpoint ratios.
    strides: Vec<usize>,
    stages: Vec<Vec<EncoderBlock>>,
    head: Conv1d,
    expansion: usize,
    pub vae_dim: usize,
}

impl TokenizerEncoder {
    /// `side` is "acoustic" or "semantic"; tensor names follow the mmproj
    /// export (`vibevoice.<side>.encoder.*`).
    pub fn from_source(
        source: &dyn TensorSource,
        side: &str,
        config: &crate::models::vibevoice_asr::config::VibeVoiceAsrConfig,
    ) -> Result<Self, String> {
        let base = format!("vibevoice.{side}.encoder");
        let strides = config.encoder_strides();
        let depths = &config.depths;
        let kernel = config.kernel_size;
        let expansion = config.ffn_expansion;
        let vae_dim = config.vae_dim_for(side)?;

        let mut downsample = Vec::with_capacity(depths.len());
        let mut layer_strides = Vec::with_capacity(depths.len());
        for stage in 0..depths.len() {
            // stage 0 is the stem: mono audio (1 channel) → n_filters;
            // later stages double: n_filters·2^(stage-1) → n_filters·2^stage
            let (n_in, n_out) = if stage == 0 {
                (1, config.n_filters)
            } else {
                (config.n_filters << (stage - 1), config.n_filters << stage)
            };
            let stride = if stage == 0 { 1 } else { strides[stage - 1] };
            let k = if stage == 0 {
                kernel
            } else {
                strides[stage - 1] * 2
            };
            downsample.push(Conv1d::from_source(
                source,
                &format!("{base}.downsample.{stage}.conv"),
                n_in,
                n_out,
                k,
            )?);
            layer_strides.push(stride);
        }

        let mut stages = Vec::with_capacity(depths.len());
        for (stage, &depth) in depths.iter().enumerate() {
            let dim = config.n_filters << stage;
            let ffn_dim = dim * expansion;
            let mut blocks = Vec::with_capacity(depth);
            for block in 0..depth {
                blocks.push(EncoderBlock::from_source(
                    source,
                    &format!("{base}.stages.{stage}.{block}"),
                    dim,
                    ffn_dim,
                    kernel,
                )?);
            }
            stages.push(blocks);
        }

        let head = Conv1d::from_source(
            source,
            &format!("{base}.head.conv"),
            config.n_filters << (depths.len() - 1),
            vae_dim,
            config.last_kernel_size,
        )?;

        Ok(Self {
            downsample,
            strides: layer_strides,
            stages,
            head,
            expansion,
            vae_dim,
        })
    }

    /// Encode token-major mono audio into token-major latent means
    /// ([t'][vae_dim]). Mirrors `TokenizerEncoder.forward` (mean only).
    pub fn forward(&self, audio: &[f32]) -> Result<Vec<f32>, String> {
        let mut traces = Vec::new();
        let out = self.forward_with_traces(audio, &mut traces)?;
        Ok(out)
    }

    /// Like `forward` but appends the token-major activations after each
    /// stage (7 entries) to `traces` for parity debugging.
    pub fn forward_with_traces(
        &self,
        audio: &[f32],
        traces: &mut Vec<Vec<f32>>,
    ) -> Result<Vec<f32>, String> {
        let mut t = audio.len();
        let mut x: Vec<f32> = audio.to_vec();
        let mut output: Vec<f32> = Vec::new();
        for stage in 0..self.stages.len() {
            let conv = &self.downsample[stage];
            let stride = self.strides[stage];
            conv.forward_token_major(&x, t, stride, &mut output);
            t = output.len() / conv.n_out;
            x.clear();
            x.extend_from_slice(&output);
            let dim = conv.n_out;
            let mut scratch = BlockScratch::new(t, dim, dim * self.expansion);
            for block in &self.stages[stage] {
                block.forward(&mut x, t, dim, &mut scratch);
            }
            traces.push(x.clone());
        }
        let mut out = Vec::new();
        self.head.forward_token_major(&x, t, 1, &mut out);
        Ok(out)
    }
}

// --------------------------------------------------------------------------- //
// SpeechConnector: fc1 → RMSNorm(eps) → fc2
// --------------------------------------------------------------------------- //

pub struct SpeechConnector {
    fc1: Dense,
    fc1_bias: Vec<f32>,
    norm: Vec<f32>,
    fc2: Dense,
    fc2_bias: Vec<f32>,
    eps: f32,
}

impl SpeechConnector {
    pub fn from_source(
        source: &dyn TensorSource,
        side: &str,
        input_dim: usize,
        output_dim: usize,
        eps: f32,
    ) -> Result<Self, String> {
        let base = format!("vibevoice.{side}.connector");
        Ok(Self {
            fc1: Dense::from_source(source, &format!("{base}.fc1"), input_dim, output_dim)?,
            fc1_bias: expect_f32_vector(source, &format!("{base}.fc1.bias"), output_dim as u64)?,
            norm: expect_f32_vector(source, &format!("{base}.norm.weight"), output_dim as u64)?,
            fc2: Dense::from_source(source, &format!("{base}.fc2"), output_dim, output_dim)?,
            fc2_bias: expect_f32_vector(source, &format!("{base}.fc2.bias"), output_dim as u64)?,
            eps,
        })
    }

    /// features: token-major [t][input_dim]; returns [t][output_dim].
    pub fn forward(
        &self,
        features: &[f32],
        t: usize,
        scratch: &mut Vec<f32>,
        output: &mut Vec<f32>,
    ) -> Result<(), String> {
        let width = self.fc1.n_out;
        if features.len() != t * self.fc1.n_in {
            return Err(format!(
                "connector input {} does not match {} frames × {}",
                features.len(),
                t,
                self.fc1.n_in
            ));
        }
        scratch.clear();
        scratch.resize(t * width, 0.0);
        self.fc1.forward_rows(features, t, &self.fc1_bias, scratch);
        for row in scratch[..t * width].chunks_exact_mut(width) {
            rms_norm_inplace(row, &self.norm, self.eps);
        }
        output.clear();
        output.resize(t * self.fc2.n_out, 0.0);
        self.fc2.forward_rows(scratch, t, &self.fc2_bias, output);
        Ok(())
    }
}
