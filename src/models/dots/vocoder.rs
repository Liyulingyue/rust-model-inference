//! AudioVAE vocoder: prompt-latent extraction encoder (`audio_encoder` +
//! `enc_mi` + `pre_proj`) and the BigVGAN decoder (`post_proj` + `dec_mi` +
//! `decoder`) that turns 128-dim latent frames into 48 kHz mono audio.
//!
//! Weight-norm was folded into plain `.weight` at export time; the fixed
//! kaiser filters of the AMP-block activations were emitted as
//! `...activations.{a}.{up,down}_filter`; the post activation keeps its
//! trained filters.
//!
//! Learned operators use native Weight kernels. Their reduction and activation
//! rounding follows the weight format; the opt-in Torch bit fixtures below
//! remain diagnostics for the former arithmetic contract.

use crate::core::tensor::TensorSource;
use crate::models::dots::patch_encoder::load_f16_f32;
use crate::ops::kernel::Weight;

use super::weights::{linear_forward, load_weight};

const LEAKY: f32 = 0.2;
const RESSTACK_LEAKY: f32 = 0.01;
const SNAKE_EPS: f32 = 1e-9;
const HOP: usize = 1920; // product of decoder upsample rates

// ---------------------------------------------------------------------------
// Convolutions retain the exported Torch row layout: [out, in, kernel] for
// Conv1d and [in, out, kernel] for ConvTranspose1d.
// ---------------------------------------------------------------------------

fn conv1d(
    weight: &Weight<'_>,
    bias: &[f32],
    input: &[f32],
    in_ch: usize,
    length: usize,
    out_ch: usize,
    kernel: usize,
    stride: usize,
    dilation: usize,
    left_pad: usize,
    out_len: usize,
) -> Vec<f32> {
    let reduction = in_ch * kernel;
    debug_assert_eq!((weight.n_in, weight.n_out), (reduction, out_ch));
    debug_assert_eq!(input.len(), in_ch * length);
    let mut output = vec![0.0; out_ch * out_len];
    // Bound im2col scratch independently of waveform duration. The batched
    // kernel reuses its activation quantization buffers across these rows.
    let batch = out_len.min(32);
    if batch == 0 {
        return output;
    }
    let mut columns = vec![0.0; batch * reduction];
    let mut projected = vec![0.0; batch * out_ch];
    for start in (0..out_len).step_by(batch) {
        let rows = (out_len - start).min(batch);
        columns[..rows * reduction].fill(0.0);
        for row in 0..rows {
            for ic in 0..in_ch {
                for tap in 0..kernel {
                    let src =
                        ((start + row) * stride + tap * dilation) as isize - left_pad as isize;
                    if (0..length as isize).contains(&src) {
                        columns[row * reduction + ic * kernel + tap] =
                            input[ic * length + src as usize];
                    }
                }
            }
        }
        linear_forward(
            weight,
            Some(bias),
            &columns[..rows * reduction],
            reduction,
            out_ch,
            &mut projected[..rows * out_ch],
        );
        for row in 0..rows {
            for oc in 0..out_ch {
                output[oc * out_len + start + row] = projected[row * out_ch + oc];
            }
        }
    }
    output
}

fn conv1d_causal(
    weight: &Weight<'_>,
    bias: &[f32],
    input: &[f32],
    in_ch: usize,
    length: usize,
    out_ch: usize,
    kernel: usize,
    dilation: usize,
    left_pad: usize,
) -> Vec<f32> {
    conv1d(
        weight, bias, input, in_ch, length, out_ch, kernel, 1, dilation, left_pad, length,
    )
}

/// Causal Conv1d with stride: output length = (length + left_pad - kernel)/stride + 1.
fn conv1d_causal_strided(
    weight: &Weight<'_>,
    bias: &[f32],
    input: &[f32],
    in_ch: usize,
    length: usize,
    out_ch: usize,
    kernel: usize,
    stride: usize,
    left_pad: usize,
) -> Vec<f32> {
    let out_len = (length + left_pad).saturating_sub(kernel) / stride + 1;
    conv1d(
        weight, bias, input, in_ch, length, out_ch, kernel, stride, 1, left_pad, out_len,
    )
}

fn conv1d_pad2(
    weight: &Weight<'_>,
    bias: &[f32],
    input: &[f32],
    in_ch: usize,
    length: usize,
    out_ch: usize,
    kernel: usize,
    pad: usize,
) -> Vec<f32> {
    conv1d(
        weight, bias, input, in_ch, length, out_ch, kernel, 1, 1, pad, length,
    )
}

/// Causal ConvTranspose1d: keep the first input_length * stride samples.
fn conv_transpose1d_causal(
    weight: &Weight<'_>,
    bias: &[f32],
    input: &[f32],
    in_ch: usize,
    length: usize,
    out_ch: usize,
    kernel: usize,
    stride: usize,
) -> Vec<f32> {
    debug_assert_eq!((weight.n_in, weight.n_out), (out_ch * kernel, in_ch));
    debug_assert_eq!(input.len(), in_ch * length);
    let out_len = length * stride;
    let mut output = vec![0.0; out_ch * out_len];
    // Only one input-channel row is expanded, including for Q8_0 weights.
    let mut row = vec![0.0; out_ch * kernel];
    for ic in 0..in_ch {
        weight.embedding_lookup(ic as u32, &mut row);
        for oc in 0..out_ch {
            for i in 0..length {
                let start = i * stride;
                let taps = kernel.min(out_len - start);
                crate::ops::vec_mad_f32(
                    &mut output[oc * out_len + start..oc * out_len + start + taps],
                    &row[oc * kernel..oc * kernel + taps],
                    input[ic * length + i],
                );
            }
        }
    }
    for oc in 0..out_ch {
        for value in &mut output[oc * out_len..(oc + 1) * out_len] {
            *value += bias[oc];
        }
    }
    output
}

fn leaky_inplace(x: &mut [f32]) {
    for value in x.iter_mut() {
        *value = if *value > 0.0 { *value } else { LEAKY * *value };
    }
}

/// SnakeBeta with logscale parameters.
fn snakebeta(x: f32, alpha: f32, beta: f32) -> f32 {
    let (a, b) = (
        super::speaker::exp::torch28_exp(alpha),
        super::speaker::exp::torch28_exp(beta),
    );
    x + (1.0 / (b + SNAKE_EPS)) * crate::ops::rope_sin_cos_sleef(x * a).1.powi(2)
}

// ---------------------------------------------------------------------------
// MI layers (Linear + skip-LSTM + Linear)
// ---------------------------------------------------------------------------

pub struct MiLayer<'a> {
    pub lin0_w: Weight<'a>,
    pub lin0_b: Vec<f32>,
    /// l = 0..4: (weight_ih [2048,512], weight_hh, bias_ih, bias_hh)
    pub lstm: [(Weight<'a>, Weight<'a>, Vec<f32>, Vec<f32>); 4],
    pub lin2_w: Weight<'a>,
    pub lin2_b: Vec<f32>,
}

fn linear_affine(
    weight: &Weight<'_>,
    bias: &[f32],
    input: &[f32],
    rows: usize,
    input_features: usize,
    output_features: usize,
) -> Vec<f32> {
    debug_assert_eq!(input.len(), rows * input_features);
    let mut output = vec![0.0; rows * output_features];
    linear_forward(
        weight,
        Some(bias),
        input,
        input_features,
        output_features,
        &mut output,
    );
    output
}

fn linear_affine_transposed_input(
    weight: &Weight<'_>,
    bias: &[f32],
    input: &[f32],
    rows: usize,
    input_features: usize,
    output_features: usize,
) -> Vec<f32> {
    debug_assert_eq!(input.len(), rows * input_features);
    let mut time_major = vec![0.0; input.len()];
    for row in 0..rows {
        for channel in 0..input_features {
            time_major[row * input_features + channel] = input[channel * rows + row];
        }
    }
    linear_affine(
        weight,
        bias,
        &time_major,
        rows,
        input_features,
        output_features,
    )
}

fn lstm_gate_row(
    hidden_weight: &Weight<'_>,
    hidden_bias: &[f32],
    hidden: &[f32],
    input_gates: &[f32],
) -> Vec<f32> {
    let mut gates = linear_affine(hidden_weight, hidden_bias, hidden, 1, 512, 2048);
    for (gate, input_gate) in gates.iter_mut().zip(input_gates) {
        *gate += input_gate;
    }
    gates
}

fn lstm_layer_forward(
    input_weight: &Weight<'_>,
    hidden_weight: &Weight<'_>,
    input_bias: &[f32],
    hidden_bias: &[f32],
    input: &[f32],
    frames: usize,
) -> Vec<f32> {
    let input_gates = linear_affine(input_weight, input_bias, input, frames, 512, 2048);
    let mut output = vec![0.0f32; frames * 512];
    let mut cell = vec![0.0f32; 512];
    let mut hidden = vec![0.0f32; 512];
    for frame in 0..frames {
        let gates = lstm_gate_row(
            hidden_weight,
            hidden_bias,
            &hidden,
            &input_gates[frame * 2048..(frame + 1) * 2048],
        );
        for channel in 0..512 {
            let input_gate = sigmoid(gates[channel]);
            let forget_gate = sigmoid(gates[512 + channel]);
            let cell_gate = tanh(gates[1024 + channel]);
            let output_gate = sigmoid(gates[1536 + channel]);
            cell[channel] = forget_gate * cell[channel] + input_gate * cell_gate;
            hidden[channel] = output_gate * tanh(cell[channel]);
            output[frame * 512 + channel] = hidden[channel];
        }
    }
    output
}

fn add_residual_in_place(input: &mut [f32], residual: &[f32]) {
    debug_assert_eq!(input.len(), residual.len());
    for (value, residual) in input.iter_mut().zip(residual) {
        *value += residual;
    }
}

fn linear_affine_channel_major(
    weight: &Weight<'_>,
    bias: &[f32],
    input: &[f32],
    rows: usize,
    input_features: usize,
    output_features: usize,
) -> Vec<f32> {
    let time_major = linear_affine(weight, bias, input, rows, input_features, output_features);
    let mut output = vec![0.0; rows * output_features];
    for row in 0..rows {
        for channel in 0..output_features {
            output[channel * rows + row] = time_major[row * output_features + channel];
        }
    }
    output
}

impl<'a> MiLayer<'a> {
    fn from_source(source: &'a dyn TensorSource, prefix: &str) -> Result<Self, String> {
        let mut lstm = Vec::with_capacity(4);
        for l in 0..4 {
            lstm.push((
                load_weight(
                    source,
                    &format!("{prefix}.1.lstm.weight_ih_l{l}"),
                    &[512, 2048],
                )?,
                load_weight(
                    source,
                    &format!("{prefix}.1.lstm.weight_hh_l{l}"),
                    &[512, 2048],
                )?,
                load_f16_f32(source, &format!("{prefix}.1.lstm.bias_ih_l{l}"), &[2048])?,
                load_f16_f32(source, &format!("{prefix}.1.lstm.bias_hh_l{l}"), &[2048])?,
            ));
        }
        Ok(Self {
            lin0_w: load_weight(source, &format!("{prefix}.0.weight"), &[128, 512])?,
            lin0_b: load_f16_f32(source, &format!("{prefix}.0.bias"), &[512])?,
            lstm: lstm
                .try_into()
                .map_err(|_| "expected four vocoder LSTM layers")?,
            lin2_w: load_weight(source, &format!("{prefix}.2.weight"), &[512, 128])?,
            lin2_b: load_f16_f32(source, &format!("{prefix}.2.bias"), &[128])?,
        })
    }

    /// `x` is `[frames, 128]`; returns `[frames, 128]`.
    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let frames = x.len() / 128;
        let h = linear_affine(&self.lin0_w, &self.lin0_b, x, frames, 128, 512);
        self.forward_after_linear0(h, frames, true)
    }

    /// `x` is the channel-major storage behind a Torch `[1, frames, 128]`
    /// transpose view; returns contiguous `[frames, 128]`.
    fn forward_transposed_input(&self, x: &[f32], frames: usize, trace_internal: bool) -> Vec<f32> {
        let h = linear_affine_transposed_input(&self.lin0_w, &self.lin0_b, x, frames, 128, 512);
        self.forward_after_linear0(h, frames, trace_internal)
    }

    fn forward_after_linear0(&self, h: Vec<f32>, frames: usize, trace_internal: bool) -> Vec<f32> {
        #[cfg(feature = "parity-trace")]
        if trace_internal {
            crate::parity_trace::report(crate::parity_trace::checkpoint(
                "dots.vocoder.dec_mi.linear0",
                None,
                &[1, frames, 512],
                &h,
            ));
        }
        // LSTM: 4 layers, gate order i/f/g/o
        let mut layer_in = h.clone();
        for layer in 0..4 {
            let (w_ih, w_hh, b_ih, b_hh) = &self.lstm[layer];
            // GGUF [512, 2048]: rows are gates in i/f/g/o order.
            layer_in = lstm_layer_forward(w_ih, w_hh, b_ih, b_hh, &layer_in, frames);
        }
        #[cfg(feature = "parity-trace")]
        if trace_internal {
            crate::parity_trace::report(crate::parity_trace::checkpoint(
                "dots.vocoder.dec_mi.lstm",
                None,
                &[1, frames, 512],
                &layer_in,
            ));
        }
        if !self.lstm.is_empty() {
            // skip connection: h + residual
            add_residual_in_place(&mut layer_in, &h);
        }
        #[cfg(feature = "parity-trace")]
        if trace_internal {
            crate::parity_trace::report(crate::parity_trace::checkpoint(
                "dots.vocoder.dec_mi.residual",
                None,
                &[1, frames, 512],
                &layer_in,
            ));
        }
        let output = linear_affine(&self.lin2_w, &self.lin2_b, &layer_in, frames, 512, 128);
        #[cfg(feature = "parity-trace")]
        if trace_internal {
            crate::parity_trace::report(crate::parity_trace::checkpoint(
                "dots.vocoder.dec_mi.linear2",
                None,
                &[1, frames, 128],
                &output,
            ));
        }
        #[cfg(not(feature = "parity-trace"))]
        let _ = trace_internal;
        output
    }
}

fn sigmoid(x: f32) -> f32 {
    super::speaker::exp::torch28_sigmoid(x)
}

#[inline(always)]
fn tanh(x: f32) -> f32 {
    super::speaker::exp::torch28_tanh(x)
}

// ---------------------------------------------------------------------------
// AudioVAE encoder (prompt latent extraction; causal)
// ---------------------------------------------------------------------------

struct EncConv<'a> {
    weight: Weight<'a>,
    bias: Vec<f32>,
    kernel: usize,
    stride: usize,
    out_ch: usize,
}

struct EncResStackLayer<'a> {
    c1: Weight<'a>,
    b1: Vec<f32>,
    d1: usize,
    c2: Weight<'a>,
    b2: Vec<f32>,
}

struct EncResStack<'a> {
    layers: Vec<EncResStackLayer<'a>>,
    ch: usize,
}

pub struct AudioEncoder<'a> {
    convs: Vec<EncConv<'a>>, // 8 convs: [pre(1→12), 6 down, post(768→128)]
    resstacks: Vec<EncResStack<'a>>, // 6
}

impl<'a> AudioEncoder<'a> {
    fn from_source(source: &'a dyn TensorSource) -> Result<Self, String> {
        let w = |name: &str, dims: &[u64]| -> Result<Vec<f32>, String> {
            load_f16_f32(source, name, dims)
        };
        let conv = |idx: usize,
                    in_ch: usize,
                    out_ch: usize,
                    kernel: usize,
                    stride: usize|
         -> Result<EncConv<'a>, String> {
            Ok(EncConv {
                weight: load_weight(
                    source,
                    &format!("dotstts.vocoder.audio_encoder.generator.{idx}.layer.weight"),
                    &[kernel as u64, in_ch as u64, out_ch as u64],
                )?,
                bias: w(
                    &format!("dotstts.vocoder.audio_encoder.generator.{idx}.layer.bias"),
                    &[out_ch as u64],
                )?,
                kernel,
                stride,
                out_ch,
            })
        };
        // downsample_rates [2,2,2,4,6,10] → channels [12,24,48,96,192,384,768]
        let convs = vec![
            conv(0, 1, 12, 3, 1)?,
            conv(2, 12, 24, 4, 2)?,
            conv(5, 24, 48, 4, 2)?,
            conv(8, 48, 96, 4, 2)?,
            conv(11, 96, 192, 8, 4)?,
            conv(14, 192, 384, 12, 6)?,
            conv(17, 384, 768, 20, 10)?,
            conv(20, 768, 128, 5, 1)?,
        ];
        let mut resstacks = Vec::new();
        let res_idx = [3usize, 6, 9, 12, 15, 18];
        for (ri, &gi) in res_idx.iter().enumerate() {
            let ch = [24usize, 48, 96, 192, 384, 768][ri];
            let mut layers = Vec::new();
            for j in 0..6 {
                let d = 1usize << j;
                layers.push(EncResStackLayer {
                    c1: load_weight(
                        source,
                        &format!(
                            "dotstts.vocoder.audio_encoder.generator.{gi}.layers.{j}.2.weight"
                        ),
                        &[3, ch as u64, ch as u64],
                    )?,
                    b1: w(
                        &format!("dotstts.vocoder.audio_encoder.generator.{gi}.layers.{j}.2.bias"),
                        &[ch as u64],
                    )?,
                    d1: d,
                    c2: load_weight(
                        source,
                        &format!(
                            "dotstts.vocoder.audio_encoder.generator.{gi}.layers.{j}.5.weight"
                        ),
                        &[3, ch as u64, ch as u64],
                    )?,
                    b2: w(
                        &format!("dotstts.vocoder.audio_encoder.generator.{gi}.layers.{j}.5.bias"),
                        &[ch as u64],
                    )?,
                });
            }
            resstacks.push(EncResStack { layers, ch });
        }
        Ok(Self { convs, resstacks })
    }

    fn forward(&self, x: &[f32]) -> Vec<f32> {
        // x [1, N] → channel-major; reference order per stage:
        //   [pre, Leaky] + Σ_stage [down-conv, ResStack, Leaky] + [post]
        let mut out = vec![0.0f32; x.len()];
        out.copy_from_slice(x);
        let mut length = x.len();
        for (ci, conv) in self.convs.iter().enumerate() {
            let in_ch = if ci == 0 {
                1
            } else {
                self.convs[ci - 1].out_ch
            };
            out = if ci == 7 {
                conv1d_pad2(
                    &conv.weight,
                    &conv.bias,
                    &out,
                    in_ch,
                    length,
                    conv.out_ch,
                    conv.kernel,
                    2,
                )
            } else {
                let left_pad = conv.kernel - 1; // causal, dilation 1
                conv1d_causal_strided(
                    &conv.weight,
                    &conv.bias,
                    &out,
                    in_ch,
                    length,
                    conv.out_ch,
                    conv.kernel,
                    conv.stride,
                    left_pad,
                )
            };
            length = out.len() / conv.out_ch;
            if ci == 0 {
                leaky_inplace(&mut out);
            } else if ci < 7 {
                let rs = &self.resstacks[ci - 1];
                out = self.resstack(rs, &out, length);
                leaky_inplace(&mut out);
            }
            // post conv (ci == 7): no activation after
        }
        out
    }

    fn resstack(&self, rs: &EncResStack, x: &[f32], length: usize) -> Vec<f32> {
        let mut cur = x.to_vec();
        for layer in &rs.layers {
            let mut h = cur.clone();
            for value in h.iter_mut() {
                *value = if *value > 0.0 {
                    *value
                } else {
                    RESSTACK_LEAKY * *value
                };
            }
            // conv1 d=layer.d1 causal pad d*(k-1)=2*d
            let pad1 = 2 * layer.d1;
            h = conv1d_causal(
                &layer.c1, &layer.b1, &h, rs.ch, length, rs.ch, 3, layer.d1, pad1,
            );
            for value in h.iter_mut() {
                *value = if *value > 0.0 {
                    *value
                } else {
                    RESSTACK_LEAKY * *value
                };
            }
            // conv2 d=1 causal pad 2
            h = conv1d_causal(&layer.c2, &layer.b2, &h, rs.ch, length, rs.ch, 3, 1, 2);
            for (a, b) in cur.iter_mut().zip(h.iter()) {
                *a += b;
            }
        }
        cur
    }
}

// ---------------------------------------------------------------------------
// BigVGAN decoder
// ---------------------------------------------------------------------------

struct AmpConv<'a> {
    weight: Weight<'a>,
    bias: Vec<f32>,
    kernel: usize,
    dilation: usize,
}

pub(crate) struct AmpBlock<'a> {
    convs1: Vec<AmpConv<'a>>,
    convs2: Vec<AmpConv<'a>>,
    alphas: Vec<f32>, // 6 [ch]
    betas: Vec<f32>,  // 6 [ch]
    ch: usize,
    up_filter: Vec<f32>,   // fixed kaiser [12]
    down_filter: Vec<f32>, // fixed kaiser [12]
}

impl<'a> AmpBlock<'a> {
    fn from_source(
        source: &'a dyn TensorSource,
        idx: usize,
        ch: usize,
        kernel: usize,
    ) -> Result<Self, String> {
        let w = |name: &str, dims: &[u64]| -> Result<Vec<f32>, String> {
            load_f16_f32(source, name, dims)
        };
        let conv = |group: &str, j: usize, dilation: usize| -> Result<AmpConv<'a>, String> {
            Ok(AmpConv {
                weight: load_weight(
                    source,
                    &format!("dotstts.vocoder.decoder.resblocks.{idx}.{group}.{j}.weight"),
                    &[kernel as u64, ch as u64, ch as u64],
                )?,
                bias: w(
                    &format!("dotstts.vocoder.decoder.resblocks.{idx}.{group}.{j}.bias"),
                    &[ch as u64],
                )?,
                kernel,
                dilation,
            })
        };
        let dilations = [1usize, 3, 5];
        let mut convs1 = Vec::new();
        let mut convs2 = Vec::new();
        for j in 0..3 {
            convs1.push(conv("convs1", j, dilations[j])?);
            convs2.push(conv("convs2", j, 1)?);
        }
        let mut alphas = Vec::new();
        let mut betas = Vec::new();
        for a in 0..6 {
            alphas.extend(w(
                &format!("dotstts.vocoder.decoder.resblocks.{idx}.activations.{a}.act.alpha"),
                &[ch as u64],
            )?);
            betas.extend(w(
                &format!("dotstts.vocoder.decoder.resblocks.{idx}.activations.{a}.act.beta"),
                &[ch as u64],
            )?);
        }
        let up_filter = w(
            &format!("dotstts.vocoder.decoder.resblocks.{idx}.activations.0.upsample.filter"),
            &[12, 1, 1],
        )?;
        let down_filter = w(
            &format!(
                "dotstts.vocoder.decoder.resblocks.{idx}.activations.0.downsample.lowpass.filter"
            ),
            &[12, 1, 1],
        )?;
        Ok(Self {
            convs1,
            convs2,
            alphas,
            betas,
            ch,
            up_filter,
            down_filter,
        })
    }

    fn forward(&self, x: &[f32], length: usize, trace_internal: bool) -> Vec<f32> {
        let mut cur = x.to_vec();
        for j in 0..3 {
            // act a[2j] → conv1 → act a[2j+1] → conv2 → residual
            let residual = cur;
            cur = self.act_j(residual.clone(), length, 2 * j, trace_internal);
            #[cfg(feature = "parity-trace")]
            if trace_internal {
                crate::parity_trace::report(crate::parity_trace::checkpoint(
                    "dots.vocoder.decoder.resblock0.act1",
                    None,
                    &[1, self.ch, length],
                    &cur,
                ));
            }
            let c1 = &self.convs1[j];
            cur = conv1d_causal(
                &c1.weight,
                &c1.bias,
                &cur,
                self.ch,
                length,
                self.ch,
                c1.kernel,
                c1.dilation,
                c1.dilation * (c1.kernel - 1),
            );
            #[cfg(feature = "parity-trace")]
            if trace_internal {
                crate::parity_trace::report(crate::parity_trace::checkpoint(
                    "dots.vocoder.decoder.resblock0.conv1",
                    None,
                    &[1, self.ch, length],
                    &cur,
                ));
            }
            cur = self.act_j(cur, length, 2 * j + 1, false);
            #[cfg(feature = "parity-trace")]
            if trace_internal {
                crate::parity_trace::report(crate::parity_trace::checkpoint(
                    "dots.vocoder.decoder.resblock0.act2",
                    None,
                    &[1, self.ch, length],
                    &cur,
                ));
            }
            let c2 = &self.convs2[j];
            cur = conv1d_causal(
                &c2.weight,
                &c2.bias,
                &cur,
                self.ch,
                length,
                self.ch,
                c2.kernel,
                1,
                c2.kernel - 1,
            );
            #[cfg(feature = "parity-trace")]
            if trace_internal {
                crate::parity_trace::report(crate::parity_trace::checkpoint(
                    "dots.vocoder.decoder.resblock0.conv2",
                    None,
                    &[1, self.ch, length],
                    &cur,
                ));
            }
            for (a, b) in cur.iter_mut().zip(residual.iter()) {
                *a += b;
            }
            #[cfg(feature = "parity-trace")]
            if trace_internal {
                crate::parity_trace::report(crate::parity_trace::checkpoint(
                    "dots.vocoder.decoder.resblock0.residual",
                    None,
                    &[1, self.ch, length],
                    &cur,
                ));
            }
        }
        #[cfg(not(feature = "parity-trace"))]
        let _ = trace_internal;
        cur
    }

    fn act_j(&self, x: Vec<f32>, length: usize, a: usize, trace_internal: bool) -> Vec<f32> {
        // Activation1d: upsample (fixed filter) → snakebeta → downsample
        let up_len = 2 * length + 11;
        let mut up = vec![0.0f32; self.ch * up_len];
        for c in 0..self.ch {
            for k in 0..12usize {
                for i in 0..length {
                    let val = x[c * length + i];
                    if val == 0.0 {
                        continue;
                    }
                    // reference multiplies the transposed conv by ratio=2
                    up[c * up_len + i * 2 + k] += 2.0 * self.up_filter[k] * val;
                }
            }
        }
        #[cfg(feature = "parity-trace")]
        if trace_internal {
            let mut trimmed = vec![0.0f32; self.ch * 2 * length];
            for c in 0..self.ch {
                trimmed[c * 2 * length..(c + 1) * 2 * length]
                    .copy_from_slice(&up[c * up_len..c * up_len + 2 * length]);
            }
            crate::parity_trace::report(crate::parity_trace::checkpoint(
                "dots.vocoder.decoder.resblock0.activation0.upsample",
                None,
                &[1, self.ch, 2 * length],
                &trimmed,
            ));
        }
        // trim: keep first 2*length
        let mut snake = vec![0.0f32; self.ch * 2 * length];
        for c in 0..self.ch {
            let alpha = self.alphas[a * self.ch + c];
            let beta = self.betas[a * self.ch + c];
            for n in 0..2 * length {
                snake[c * 2 * length + n] = snakebeta(up[c * up_len + n], alpha, beta);
            }
        }
        #[cfg(feature = "parity-trace")]
        if trace_internal {
            crate::parity_trace::report(crate::parity_trace::checkpoint(
                "dots.vocoder.decoder.resblock0.activation0.snakebeta",
                None,
                &[1, self.ch, 2 * length],
                &snake,
            ));
        }
        // downsample: replicate pad 11 left, conv1d stride 2
        let mut down = vec![0.0f32; self.ch * length];
        for c in 0..self.ch {
            for n in 0..length {
                let mut acc = 0.0f32;
                for k in 0..12usize {
                    let src = n * 2 + k;
                    let padded = if src >= 11 {
                        src - 11
                    } else {
                        0 // replicate: index 0
                    };
                    acc += self.down_filter[k] * snake[c * 2 * length + padded];
                }
                down[c * length + n] = acc;
            }
        }
        #[cfg(not(feature = "parity-trace"))]
        let _ = trace_internal;
        down
    }
}

pub struct BigVganDecoder<'a> {
    pub conv_pre: (Weight<'a>, Vec<f32>), // [5,128,1536]
    pub ups: Vec<(Weight<'a>, Vec<f32>, usize, usize, usize, usize)>, // w,b,kernel,stride,in,out
    pub(crate) resblocks: Vec<AmpBlock<'a>>,
    pub post_alpha: Vec<f32>,
    pub post_beta: Vec<f32>,
    pub post_up: Vec<f32>, // trained [24,1,12] → flattened per-channel [24*12]
    pub post_down: Vec<f32>, // trained [24,1,12]
    pub conv_post: (Weight<'a>, Vec<f32>), // [7,24,1]
}

impl<'a> BigVganDecoder<'a> {
    fn from_source(source: &'a dyn TensorSource) -> Result<Self, String> {
        let w = |name: &str, dims: &[u64]| -> Result<Vec<f32>, String> {
            load_f16_f32(source, name, dims)
        };
        let conv_pre = (
            load_weight(
                source,
                "dotstts.vocoder.decoder.conv_pre.weight",
                &[5, 128, 1536],
            )?,
            w("dotstts.vocoder.decoder.conv_pre.bias", &[1536])?,
        );
        let mut ups = Vec::new();
        let spec = [
            (20usize, 10usize, 1536usize, 768usize),
            (12, 6, 768, 384),
            (8, 4, 384, 192),
            (4, 2, 192, 96),
            (4, 2, 96, 48),
            (4, 2, 48, 24),
        ];
        for (i, &(k, s, ich, och)) in spec.iter().enumerate() {
            ups.push((
                load_weight(
                    source,
                    &format!("dotstts.vocoder.decoder.ups.{i}.0.weight"),
                    &[k as u64, och as u64, ich as u64],
                )?,
                w(
                    &format!("dotstts.vocoder.decoder.ups.{i}.0.bias"),
                    &[och as u64],
                )?,
                k,
                s,
                ich,
                och,
            ));
        }
        let mut resblocks = Vec::new();
        let stage_kernels = [3usize, 7, 11];
        for stage in 0..6 {
            let ch = [768usize, 384, 192, 96, 48, 24][stage];
            for j in 0..3 {
                let idx = stage * 3 + j;
                resblocks.push(AmpBlock::from_source(source, idx, ch, stage_kernels[j])?);
            }
        }
        let post_alpha = w("dotstts.vocoder.decoder.activation_post.act.alpha", &[24])?;
        let post_beta = w("dotstts.vocoder.decoder.activation_post.act.beta", &[24])?;
        // trained filters: gguf dims [12,1,24] → [ch][k]
        let post_up_raw = w(
            "dotstts.vocoder.decoder.activation_post.upsample.filter",
            &[12, 1, 24],
        )?;
        let post_down_raw = w(
            "dotstts.vocoder.decoder.activation_post.downsample.lowpass.filter",
            &[12, 1, 24],
        )?;
        let mut post_up = vec![0.0f32; 24 * 12];
        let mut post_down = vec![0.0f32; 24 * 12];
        for c in 0..24 {
            for k in 0..12 {
                post_up[c * 12 + k] = post_up_raw[c * 12 + k];
                post_down[c * 12 + k] = post_down_raw[c * 12 + k];
            }
        }
        let conv_post = (
            load_weight(
                source,
                "dotstts.vocoder.decoder.conv_post.weight",
                &[7, 24, 1],
            )?,
            // The checkpoint may omit this bias; malformed stored biases
            // must still fail validation.
            if source
                .tensor_info("dotstts.vocoder.decoder.conv_post.bias")
                .is_some()
            {
                w("dotstts.vocoder.decoder.conv_post.bias", &[1])?
            } else {
                vec![0.0]
            },
        );
        Ok(Self {
            conv_pre,
            ups,
            resblocks,
            post_alpha,
            post_beta,
            post_up,
            post_down,
            conv_post,
        })
    }

    fn forward(&self, x: &[f32]) -> Vec<f32> {
        // x: [128, T]
        let mut length = x.len() / 128;
        let mut cur = conv1d_pad2(
            &self.conv_pre.0,
            &self.conv_pre.1,
            x,
            128,
            length,
            1536,
            5,
            2,
        );
        #[cfg(feature = "parity-trace")]
        crate::parity_trace::report(crate::parity_trace::checkpoint(
            "dots.vocoder.decoder.conv_pre",
            None,
            &[1, 1536, length],
            &cur,
        ));
        for i in 0..6 {
            let (w_, b, kernel, stride, in_ch, out_ch) = &self.ups[i];
            cur = conv_transpose1d_causal(w_, b, &cur, *in_ch, length, *out_ch, *kernel, *stride);
            length *= stride;
            #[cfg(feature = "parity-trace")]
            crate::parity_trace::report(crate::parity_trace::checkpoint(
                "dots.vocoder.decoder.up",
                None,
                &[1, *out_ch, length],
                &cur,
            ));
            // 3 AMP blocks summed
            let mut acc = self.resblocks[i * 3].forward(&cur, length, i == 0);
            #[cfg(feature = "parity-trace")]
            crate::parity_trace::report(crate::parity_trace::checkpoint(
                "dots.vocoder.decoder.resblock",
                None,
                &[1, *out_ch, length],
                &acc,
            ));
            let sum2 = self.resblocks[i * 3 + 1].forward(&cur, length, false);
            #[cfg(feature = "parity-trace")]
            crate::parity_trace::report(crate::parity_trace::checkpoint(
                "dots.vocoder.decoder.resblock",
                None,
                &[1, *out_ch, length],
                &sum2,
            ));
            for (a, b) in acc.iter_mut().zip(sum2.iter()) {
                *a += b;
            }
            let sum3 = self.resblocks[i * 3 + 2].forward(&cur, length, false);
            #[cfg(feature = "parity-trace")]
            crate::parity_trace::report(crate::parity_trace::checkpoint(
                "dots.vocoder.decoder.resblock",
                None,
                &[1, *out_ch, length],
                &sum3,
            ));
            for (a, b) in acc.iter_mut().zip(sum3.iter()) {
                *a += b;
            }
            for value in acc.iter_mut() {
                *value /= 3.0;
            }
            cur = acc;
            #[cfg(feature = "parity-trace")]
            crate::parity_trace::report(crate::parity_trace::checkpoint(
                "dots.vocoder.decoder.stage",
                None,
                &[1, *out_ch, length],
                &cur,
            ));
        }
        // activation_post (trained filters) + conv_post + clamp
        let ch = 24usize;
        let up_len = 2 * length + 11;
        let mut up = vec![0.0f32; ch * up_len];
        for c in 0..ch {
            for k in 0..12usize {
                for i in 0..length {
                    let val = cur[c * length + i];
                    if val == 0.0 {
                        continue;
                    }
                    // reference multiplies the transposed conv by ratio=2
                    up[c * up_len + i * 2 + k] += 2.0 * self.post_up[c * 12 + k] * val;
                }
            }
        }
        #[cfg(feature = "parity-trace")]
        {
            let mut trimmed = vec![0.0f32; ch * 2 * length];
            for c in 0..ch {
                trimmed[c * 2 * length..(c + 1) * 2 * length]
                    .copy_from_slice(&up[c * up_len..c * up_len + 2 * length]);
            }
            crate::parity_trace::report(crate::parity_trace::checkpoint(
                "dots.vocoder.decoder.activation_post.upsample",
                None,
                &[1, ch, 2 * length],
                &trimmed,
            ));
        }
        let mut snake = vec![0.0f32; ch * 2 * length];
        for c in 0..ch {
            let (alpha, beta) = (self.post_alpha[c], self.post_beta[c]);
            for n in 0..2 * length {
                snake[c * 2 * length + n] = snakebeta(up[c * up_len + n], alpha, beta);
            }
        }
        #[cfg(feature = "parity-trace")]
        crate::parity_trace::report(crate::parity_trace::checkpoint(
            "dots.vocoder.decoder.activation_post.snakebeta",
            None,
            &[1, ch, 2 * length],
            &snake,
        ));
        let mut down = vec![0.0f32; ch * length];
        for c in 0..ch {
            for n in 0..length {
                let mut acc = 0.0f32;
                for k in 0..12usize {
                    let src = n * 2 + k;
                    let padded = if src >= 11 { src - 11 } else { 0 };
                    acc += self.post_down[c * 12 + k] * snake[c * 2 * length + padded];
                }
                down[c * length + n] = acc;
            }
        }
        #[cfg(feature = "parity-trace")]
        crate::parity_trace::report(crate::parity_trace::checkpoint(
            "dots.vocoder.decoder.activation_post",
            None,
            &[1, ch, length],
            &down,
        ));
        // conv_post: [24, L] → [1, L] causal pad 6
        let final_len = length;
        let mut audio = conv1d_causal(
            &self.conv_post.0,
            &self.conv_post.1,
            &down,
            24,
            final_len,
            1,
            7,
            1,
            6,
        );
        for value in &mut audio {
            *value = value.clamp(-1.0, 1.0); // use_tanh_at_final=False → clamp
        }
        audio
    }
}

// ---------------------------------------------------------------------------
// Assembled vocoder
// ---------------------------------------------------------------------------

pub struct Vocoder<'a> {
    pub(crate) encoder: AudioEncoder<'a>,
    pub(crate) enc_mi: MiLayer<'a>,
    pub pre_proj: (Weight<'a>, Vec<f32>),  // [1,128,256]
    pub post_proj: (Weight<'a>, Vec<f32>), // [1,128,128]
    pub(crate) dec_mi: MiLayer<'a>,
    pub(crate) decoder: BigVganDecoder<'a>,
}

impl<'a> Vocoder<'a> {
    pub fn from_source(source: &'a dyn TensorSource) -> Result<Self, String> {
        let w = |name: &str, dims: &[u64]| -> Result<Vec<f32>, String> {
            load_f16_f32(source, name, dims)
        };
        Ok(Self {
            encoder: AudioEncoder::from_source(source)?,
            enc_mi: MiLayer::from_source(source, "dotstts.vocoder.enc_mi_layer")?,
            pre_proj: (
                load_weight(source, "dotstts.vocoder.pre_proj.weight", &[1, 128, 256])?,
                w("dotstts.vocoder.pre_proj.bias", &[256])?,
            ),
            post_proj: (
                load_weight(source, "dotstts.vocoder.post_proj.weight", &[1, 128, 128])?,
                w("dotstts.vocoder.post_proj.bias", &[128])?,
            ),
            dec_mi: MiLayer::from_source(source, "dotstts.vocoder.dec_mi_layer")?,
            decoder: BigVganDecoder::from_source(source)?,
        })
    }

    /// Encode a 48 kHz mono waveform to the latent distribution `[256, T]`
    /// (rows 0..128 = mean, rows 128..256 = log_std).
    pub fn extract_latent_distribution(&self, waveform: &[f32]) -> Result<Vec<f32>, String> {
        let encoded = self.encoder.forward(waveform); // [128, T]
        let frames = encoded.len() / 128;
        if frames == 0 {
            return Err("vocoder encoder produced no frames".into());
        }
        // Convert the channel-major encoder output for the first MI Linear.
        let mi_out = self
            .enc_mi
            .forward_transposed_input(&encoded, frames, false); // [T, 128]
                                                                // permute back to [128, T], then pre_proj 1x1 conv (128 → 256)
        Ok(linear_affine_channel_major(
            &self.pre_proj.0,
            &self.pre_proj.1,
            &mi_out,
            frames,
            128,
            256,
        ))
    }

    /// Decode `[frames, 128]` raw latents → 48 kHz mono waveform.
    pub fn decode_latents(&self, latents: &[f32]) -> Result<Vec<f32>, String> {
        let frames = latents.len() / 128;
        if latents.len() != frames * 128 || frames == 0 {
            return Err("vocoder decode_latents expects [frames, 128]".into());
        }
        // post_proj (1x1, 128→128)
        let post = linear_affine_channel_major(
            &self.post_proj.0,
            &self.post_proj.1,
            latents,
            frames,
            128,
            128,
        );
        #[cfg(feature = "parity-trace")]
        crate::parity_trace::report(crate::parity_trace::checkpoint(
            "dots.vocoder.post_proj",
            None,
            &[1, 128, frames],
            &post,
        ));
        // Preserve Torch's non-contiguous permute view through the first Linear.
        let mi_out = self.dec_mi.forward_transposed_input(&post, frames, true); // [T, 128]
        #[cfg(feature = "parity-trace")]
        crate::parity_trace::report(crate::parity_trace::checkpoint(
            "dots.vocoder.dec_mi",
            None,
            &[1, frames, 128],
            &mi_out,
        ));
        let mut dec_in = vec![0.0f32; 128 * frames];
        for t in 0..frames {
            for c in 0..128 {
                dec_in[c * frames + t] = mi_out[t * 128 + c];
            }
        }
        let waveform = self.decoder.forward(&dec_in);
        #[cfg(feature = "parity-trace")]
        crate::parity_trace::report(crate::parity_trace::checkpoint(
            "dots.vocoder.waveform",
            None,
            &[waveform.len()],
            &waveform,
        ));
        Ok(waveform)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_weight(values: Vec<f32>, n_in: usize) -> Weight<'static> {
        let n_out = values.len() / n_in;
        let mut weight = Weight::from_quantized(crate::ops::kernel::QuantizedTensor::F32(values));
        weight.n_in = n_in;
        weight.n_out = n_out;
        weight
    }

    fn read_f32_path(path: impl AsRef<std::path::Path>) -> Vec<f32> {
        std::fs::read(path)
            .unwrap()
            .chunks_exact(4)
            .map(|word| f32::from_le_bytes(word.try_into().unwrap()))
            .collect()
    }

    fn channel_major_prefix(
        input: &[f32],
        channels: usize,
        full_length: usize,
        length: usize,
    ) -> Vec<f32> {
        assert_eq!(input.len(), channels * full_length);
        let mut output = Vec::with_capacity(channels * length);
        for channel in 0..channels {
            output.extend_from_slice(&input[channel * full_length..channel * full_length + length]);
        }
        output
    }

    fn assert_f32_bits_eq(actual: &[f32], expected: &[f32], label: &str) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(actual.to_bits(), expected.to_bits(), "{label}[{index}]");
        }
    }

    #[test]
    #[ignore = "requires DOTS_AUDIOENC_MMPROJ, DOTS_AUDIOENC_INPUT, and DOTS_AUDIOENC_CONV0"]
    fn audio_encoder_preconv_matches_pinned_torch_sgemm_bitwise() {
        use crate::{open_model_source, ComponentRole};

        let read_f32 = |name: &str| {
            std::fs::read(std::env::var_os(name).unwrap())
                .unwrap()
                .chunks_exact(4)
                .map(|word| f32::from_le_bytes(word.try_into().unwrap()))
                .collect::<Vec<_>>()
        };
        let path = std::path::PathBuf::from(std::env::var_os("DOTS_AUDIOENC_MMPROJ").unwrap());
        let source = open_model_source(&path, ComponentRole::Mmproj).unwrap();
        let mut input = read_f32("DOTS_AUDIOENC_INPUT");
        input.resize(284_160, 0.0);
        let weight = load_weight(
            source.as_ref(),
            "dotstts.vocoder.audio_encoder.generator.0.layer.weight",
            &[3, 1, 12],
        )
        .unwrap();
        let bias = load_f16_f32(
            source.as_ref(),
            "dotstts.vocoder.audio_encoder.generator.0.layer.bias",
            &[12],
        )
        .unwrap();
        let expected = read_f32("DOTS_AUDIOENC_CONV0");

        let actual = conv1d_causal_strided(&weight, &bias, &input, 1, 284_160, 12, 3, 1, 2);

        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "generator.0.layer[{index}]"
            );
        }
    }

    #[test]
    #[ignore = "requires DOTS_AUDIOENC_MMPROJ and DOTS_AUDIOENC_ORACLE_DIR"]
    fn audio_encoder_resstack_matches_pinned_torch_sgemm_bitwise() {
        use crate::{open_model_source, ComponentRole};

        const CHANNELS: usize = 24;
        const FULL_LENGTH: usize = 142_080;
        const PREFIX_LENGTH: usize = 256;

        let model = std::path::PathBuf::from(std::env::var_os("DOTS_AUDIOENC_MMPROJ").unwrap());
        let oracle =
            std::path::PathBuf::from(std::env::var_os("DOTS_AUDIOENC_ORACLE_DIR").unwrap());
        let source = open_model_source(&model, ComponentRole::Mmproj).unwrap();
        let encoder = AudioEncoder::from_source(source.as_ref()).unwrap();
        let input = channel_major_prefix(
            &read_f32_path(oracle.join("g2_out.f32")),
            CHANNELS,
            FULL_LENGTH,
            PREFIX_LENGTH,
        );
        let expected = channel_major_prefix(
            &read_f32_path(oracle.join("g3_resstack_out.f32")),
            CHANNELS,
            FULL_LENGTH,
            PREFIX_LENGTH,
        );

        let actual = encoder.resstack(&encoder.resstacks[0], &input, PREFIX_LENGTH);

        assert_f32_bits_eq(&actual, &expected, "generator.3");
    }

    #[test]
    #[ignore = "requires DOTS_AUDIOENC_MMPROJ and DOTS_AUDIOENC_ORACLE_DIR"]
    fn audio_encoder_postconv_matches_pinned_torch_sgemm_bitwise() {
        use crate::{open_model_source, ComponentRole};

        const CHANNELS: usize = 768;
        const LENGTH: usize = 148;

        let model = std::path::PathBuf::from(std::env::var_os("DOTS_AUDIOENC_MMPROJ").unwrap());
        let oracle =
            std::path::PathBuf::from(std::env::var_os("DOTS_AUDIOENC_ORACLE_DIR").unwrap());
        let source = open_model_source(&model, ComponentRole::Mmproj).unwrap();
        let encoder = AudioEncoder::from_source(source.as_ref()).unwrap();
        let input = read_f32_path(oracle.join("g19_out.f32"));
        let expected = read_f32_path(oracle.join("g20_out.f32"));
        let conv = &encoder.convs[7];

        assert_eq!(input.len(), CHANNELS * LENGTH);
        let actual = conv1d_pad2(
            &conv.weight,
            &conv.bias,
            &input,
            CHANNELS,
            LENGTH,
            conv.out_ch,
            conv.kernel,
            2,
        );

        assert_f32_bits_eq(&actual, &expected, "generator.20");
    }

    #[test]
    #[ignore = "requires DOTS_AUDIOENC_MMPROJ and DOTS_AUDIOENC_ORACLE_DIR"]
    fn mi_lstm_input_affine_matches_batched_pinned_torch_sgemm_bitwise() {
        use crate::{open_model_source, ComponentRole};

        const FRAMES: usize = 148;

        let model = std::path::PathBuf::from(std::env::var_os("DOTS_AUDIOENC_MMPROJ").unwrap());
        let oracle =
            std::path::PathBuf::from(std::env::var_os("DOTS_AUDIOENC_ORACLE_DIR").unwrap());
        let source = open_model_source(&model, ComponentRole::Mmproj).unwrap();
        let mi = MiLayer::from_source(source.as_ref(), "dotstts.vocoder.enc_mi_layer").unwrap();
        let input = read_f32_path(oracle.join("enc_mi_linear0.f32"));
        let expected = read_f32_path(oracle.join("l0_ih_batched_addmm.f32"));

        let actual = linear_affine(&mi.lstm[0].0, &mi.lstm[0].2, &input, FRAMES, 512, 2048);

        assert_eq!(expected[1].to_bits(), 0xbe02_304d);
        assert_f32_bits_eq(&actual, &expected, "enc_mi.lstm.0.input_affine");
    }

    #[test]
    #[ignore = "requires DOTS_AUDIOENC_MMPROJ and DOTS_AUDIOENC_ORACLE_DIR"]
    fn mi_linear0_matches_pinned_torch_sgemm_bitwise() {
        use crate::{open_model_source, ComponentRole};

        const FRAMES: usize = 148;

        let model = std::path::PathBuf::from(std::env::var_os("DOTS_AUDIOENC_MMPROJ").unwrap());
        let oracle =
            std::path::PathBuf::from(std::env::var_os("DOTS_AUDIOENC_ORACLE_DIR").unwrap());
        let source = open_model_source(&model, ComponentRole::Mmproj).unwrap();
        let mi = MiLayer::from_source(source.as_ref(), "dotstts.vocoder.enc_mi_layer").unwrap();
        let input = read_f32_path(oracle.join("enc_time.f32"));
        let expected = read_f32_path(oracle.join("enc_mi_linear0.f32"));

        let actual = linear_affine(&mi.lin0_w, &mi.lin0_b, &input, FRAMES, 128, 512);

        assert_f32_bits_eq(&actual, &expected, "enc_mi.linear0");
    }

    #[test]
    #[ignore = "requires DOTS_AUDIOENC_MMPROJ and DOTS_AUDIOENC_ORACLE_DIR"]
    fn mi_linear2_matches_pinned_torch_sgemm_bitwise() {
        use crate::{open_model_source, ComponentRole};

        const FRAMES: usize = 148;

        let model = std::path::PathBuf::from(std::env::var_os("DOTS_AUDIOENC_MMPROJ").unwrap());
        let oracle =
            std::path::PathBuf::from(std::env::var_os("DOTS_AUDIOENC_ORACLE_DIR").unwrap());
        let source = open_model_source(&model, ComponentRole::Mmproj).unwrap();
        let mi = MiLayer::from_source(source.as_ref(), "dotstts.vocoder.enc_mi_layer").unwrap();
        let input = read_f32_path(oracle.join("enc_mi_skip.f32"));
        let expected = read_f32_path(oracle.join("enc_mi_linear2.f32"));

        let actual = linear_affine(&mi.lin2_w, &mi.lin2_b, &input, FRAMES, 512, 128);

        assert_f32_bits_eq(&actual, &expected, "enc_mi.linear2");
    }

    #[test]
    #[ignore = "requires DOTS_AUDIOENC_ORACLE_DIR"]
    fn mi_skip_add_matches_pinned_torch_bitwise() {
        let oracle =
            std::path::PathBuf::from(std::env::var_os("DOTS_AUDIOENC_ORACLE_DIR").unwrap());
        let mut actual = read_f32_path(oracle.join("enc_mi_lstm_l3.f32"));
        let residual = read_f32_path(oracle.join("enc_mi_linear0.f32"));
        let expected = read_f32_path(oracle.join("enc_mi_skip.f32"));

        add_residual_in_place(&mut actual, &residual);

        assert_f32_bits_eq(&actual, &expected, "enc_mi.skip");
    }

    #[test]
    #[ignore = "requires DOTS_AUDIOENC_MMPROJ and DOTS_AUDIOENC_ORACLE_DIR"]
    fn mi_forward_matches_pinned_torch_bitwise() {
        use crate::{open_model_source, ComponentRole};

        let model = std::path::PathBuf::from(std::env::var_os("DOTS_AUDIOENC_MMPROJ").unwrap());
        let oracle =
            std::path::PathBuf::from(std::env::var_os("DOTS_AUDIOENC_ORACLE_DIR").unwrap());
        let source = open_model_source(&model, ComponentRole::Mmproj).unwrap();
        let mi = MiLayer::from_source(source.as_ref(), "dotstts.vocoder.enc_mi_layer").unwrap();
        let input = read_f32_path(oracle.join("enc_time.f32"));
        let expected = read_f32_path(oracle.join("enc_mi_linear2.f32"));

        let actual = mi.forward(&input);

        assert_f32_bits_eq(&actual, &expected, "enc_mi");
    }

    #[test]
    #[ignore = "requires DOTS_AUDIOENC_MMPROJ and DOTS_AUDIOENC_ORACLE_DIR"]
    fn pre_proj_matches_pinned_torch_sgemm_bitwise() {
        use crate::{open_model_source, ComponentRole};

        const FRAMES: usize = 148;

        let model = std::path::PathBuf::from(std::env::var_os("DOTS_AUDIOENC_MMPROJ").unwrap());
        let oracle =
            std::path::PathBuf::from(std::env::var_os("DOTS_AUDIOENC_ORACLE_DIR").unwrap());
        let source = open_model_source(&model, ComponentRole::Mmproj).unwrap();
        let weight = load_weight(
            source.as_ref(),
            "dotstts.vocoder.pre_proj.weight",
            &[1, 128, 256],
        )
        .unwrap();
        let bias = load_f16_f32(source.as_ref(), "dotstts.vocoder.pre_proj.bias", &[256]).unwrap();
        let input = read_f32_path(oracle.join("enc_mi_linear2.f32"));
        let expected = read_f32_path(oracle.join("pre_proj.f32"));

        let actual = linear_affine_channel_major(&weight, &bias, &input, FRAMES, 128, 256);

        assert_f32_bits_eq(&actual, &expected, "pre_proj");
    }

    #[test]
    #[ignore = "requires DOTS_AUDIOENC_MMPROJ, DOTS_AUDIOENC_INPUT, and DOTS_AUDIOENC_ORACLE_DIR"]
    fn prompt_distribution_matches_pinned_torch_bitwise() {
        use crate::{open_model_source, ComponentRole};

        let model = std::path::PathBuf::from(std::env::var_os("DOTS_AUDIOENC_MMPROJ").unwrap());
        let mut input = read_f32_path(std::env::var_os("DOTS_AUDIOENC_INPUT").unwrap());
        input.resize(37 * 7_680, 0.0);
        let oracle =
            std::path::PathBuf::from(std::env::var_os("DOTS_AUDIOENC_ORACLE_DIR").unwrap());
        let expected = read_f32_path(oracle.join("pre_proj.f32"));
        let source = open_model_source(&model, ComponentRole::Mmproj).unwrap();
        let vocoder = Vocoder::from_source(source.as_ref()).unwrap();

        let actual = vocoder.extract_latent_distribution(&input).unwrap();

        assert_f32_bits_eq(&actual, &expected, "dots.prompt.distribution");
    }

    #[test]
    #[ignore = "requires DOTS_AUDIOENC_MMPROJ and DOTS_AUDIOENC_ORACLE_DIR"]
    fn mi_lstm_frame0_hidden_affine_and_gate_add_match_pinned_torch_bitwise() {
        use crate::{open_model_source, ComponentRole};

        let model = std::path::PathBuf::from(std::env::var_os("DOTS_AUDIOENC_MMPROJ").unwrap());
        let oracle =
            std::path::PathBuf::from(std::env::var_os("DOTS_AUDIOENC_ORACLE_DIR").unwrap());
        let source = open_model_source(&model, ComponentRole::Mmproj).unwrap();
        let mi = MiLayer::from_source(source.as_ref(), "dotstts.vocoder.enc_mi_layer").unwrap();
        let input = read_f32_path(oracle.join("enc_mi_linear0.f32"));
        let input_gates = linear_affine(&mi.lstm[0].0, &mi.lstm[0].2, &input, 148, 512, 2048);
        let zeros = vec![0.0f32; 512];

        let hidden_gates = linear_affine(&mi.lstm[0].1, &mi.lstm[0].3, &zeros, 1, 512, 2048);
        let expected_hidden = read_f32_path(oracle.join("l0_t0_hh_addmm.f32"));
        assert_eq!(expected_hidden[0].to_bits(), 0x3d3f_e646);
        assert_f32_bits_eq(
            &hidden_gates,
            &expected_hidden,
            "enc_mi.lstm.0.hidden_affine",
        );

        let actual = lstm_gate_row(&mi.lstm[0].1, &mi.lstm[0].3, &zeros, &input_gates[..2048]);
        let expected = read_f32_path(oracle.join("l0_t0_gate_batched_add.f32"));
        assert_eq!(expected[0].to_bits(), 0xbe1c_4e52);
        assert_f32_bits_eq(&actual, &expected, "enc_mi.lstm.0.gates");
    }

    #[test]
    #[ignore = "requires DOTS_AUDIOENC_MMPROJ and DOTS_AUDIOENC_ORACLE_DIR"]
    fn mi_lstm_layers_match_pinned_torch_bitwise() {
        use crate::{open_model_source, ComponentRole};

        const FRAMES: usize = 148;

        let model = std::path::PathBuf::from(std::env::var_os("DOTS_AUDIOENC_MMPROJ").unwrap());
        let oracle =
            std::path::PathBuf::from(std::env::var_os("DOTS_AUDIOENC_ORACLE_DIR").unwrap());
        let source = open_model_source(&model, ComponentRole::Mmproj).unwrap();
        let mi = MiLayer::from_source(source.as_ref(), "dotstts.vocoder.enc_mi_layer").unwrap();
        let mut input = read_f32_path(oracle.join("enc_mi_linear0.f32"));
        for (layer, (input_weight, hidden_weight, input_bias, hidden_bias)) in
            mi.lstm.iter().enumerate()
        {
            let actual = lstm_layer_forward(
                input_weight,
                hidden_weight,
                input_bias,
                hidden_bias,
                &input,
                FRAMES,
            );
            let expected = read_f32_path(oracle.join(format!("enc_mi_lstm_l{layer}.f32")));
            assert_f32_bits_eq(&actual, &expected, &format!("enc_mi.lstm.{layer}"));
            input = actual;
        }
    }

    #[test]
    #[ignore = "requires DOTS_AUDIOENC_ORACLE_DIR"]
    fn mi_lstm_sigmoid_matches_pinned_torch_vector_kernel_bitwise() {
        let oracle =
            std::path::PathBuf::from(std::env::var_os("DOTS_AUDIOENC_ORACLE_DIR").unwrap());
        let gates = read_f32_path(oracle.join("l0_t0_gate_batched_add.f32"));
        let expected = read_f32_path(oracle.join("l0_t0_i_sigmoid_batched.f32"));

        let actual = gates[..512]
            .iter()
            .copied()
            .map(sigmoid)
            .collect::<Vec<_>>();

        assert_eq!(expected[11].to_bits(), 0x3edb_9742);
        assert_f32_bits_eq(&actual, &expected, "enc_mi.lstm.0.i_sigmoid");
    }

    #[test]
    #[ignore = "requires DOTS_AUDIOENC_ORACLE_DIR"]
    fn mi_lstm_tanh_matches_pinned_torch_vector_kernel_bitwise() {
        let oracle =
            std::path::PathBuf::from(std::env::var_os("DOTS_AUDIOENC_ORACLE_DIR").unwrap());
        let gates = read_f32_path(oracle.join("l0_t0_gate_batched_add.f32"));
        let expected = read_f32_path(oracle.join("l0_t0_g_tanh_batched.f32"));

        let actual = gates[1024..1536]
            .iter()
            .copied()
            .map(tanh)
            .collect::<Vec<_>>();

        assert_eq!(expected[52].to_bits(), 0x3f17_703c);
        assert_f32_bits_eq(&actual, &expected, "enc_mi.lstm.0.g_tanh");
    }

    #[test]
    #[ignore = "requires DOTS_AUDIOENC_TANH_ORACLE_DIR"]
    fn mi_lstm_cell_tanh_matches_pinned_torch_vector_kernel_bitwise() {
        let oracle =
            std::path::PathBuf::from(std::env::var_os("DOTS_AUDIOENC_TANH_ORACLE_DIR").unwrap());
        let input = read_f32_path(oracle.join("real_cell_input.f32"));
        let expected = read_f32_path(oracle.join("real_cell_tanh_torch.f32"));

        let actual = input.iter().copied().map(tanh).collect::<Vec<_>>();

        assert_f32_bits_eq(&actual, &expected, "enc_mi.lstm.0.cell_tanh");
    }

    #[test]
    fn encoder_resstack_applies_leaky_relu_before_each_first_convolution() {
        let stack = EncResStack {
            layers: vec![EncResStackLayer {
                c1: f32_weight(vec![0.0, 0.0, 1.0], 3),
                b1: vec![0.0],
                d1: 1,
                c2: f32_weight(vec![0.0, 0.0, 1.0], 3),
                b2: vec![0.0],
            }],
            ch: 1,
        };

        let output = AudioEncoder {
            convs: Vec::new(),
            resstacks: Vec::new(),
        }
        .resstack(&stack, &[-1.0, 1.0], 2);

        assert_eq!(output, vec![-1.0001, 2.0]);
    }

    #[test]
    fn encoder_final_convolution_uses_default_two_frame_lookahead() {
        let mut convs = (0..7)
            .map(|_| EncConv {
                weight: f32_weight(vec![1.0], 1),
                bias: vec![0.0],
                kernel: 1,
                stride: 1,
                out_ch: 1,
            })
            .collect::<Vec<_>>();
        convs.push(EncConv {
            weight: f32_weight(vec![1.0, 0.0, 0.0, 0.0, 0.0], 5),
            bias: vec![0.0],
            kernel: 5,
            stride: 1,
            out_ch: 1,
        });
        let resstacks = (0..6)
            .map(|_| EncResStack {
                layers: Vec::new(),
                ch: 1,
            })
            .collect();

        let output = AudioEncoder { convs, resstacks }.forward(&[1.0, 2.0, 3.0, 4.0, 5.0]);

        assert_eq!(output, vec![0.0, 0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn snakebeta_matches_reference_formula() {
        // y = x + sin²(x·exp(a)) / (exp(b) + 1e-9)
        let y = snakebeta(0.5, 0.0, 0.0);
        let expected = 0.5 + (0.5f32).sin().powi(2) / (1.0 + SNAKE_EPS);
        assert!((y - expected).abs() < 1e-6);
    }

    #[test]
    fn conv_transpose_causal_produces_exact_length() {
        let w = f32_weight(vec![1.0f32; 4 * 2 * 4], 2 * 4); // in=4,out=2,k=4
        let b = vec![0.0f32; 2];
        let x = vec![1.0f32; 4 * 3]; // 3 frames
        let y = conv_transpose1d_causal(&w, &b, &x, 4, 3, 2, 4, 2);
        assert_eq!(y.len(), 2 * 3 * 2);
    }

    fn q8_fixture(n_in: usize, n_out: usize) -> (Vec<u8>, Vec<f32>) {
        assert_eq!(n_in % 32, 0);
        let mut bytes = Vec::new();
        let mut values = Vec::new();
        for block in 0..n_in * n_out / 32 {
            let scale = (block % 4 + 1) as f32 / 64.0;
            bytes.extend_from_slice(&crate::ops::f32_to_f16(scale).to_le_bytes());
            for lane in 0..32 {
                let q = ((block * 7 + lane * 3) % 23) as i8 - 11;
                bytes.push(q as u8);
                values.push(q as f32 * scale);
            }
        }
        (bytes, values)
    }

    #[test]
    fn q8_convolution_matches_scalar_layout_bias_stride_dilation_and_batch_tail() {
        use crate::core::tensor::GGMLType;
        use crate::ops::kernel::QuantizedTensor;
        let (in_ch, out_ch, length, kernel) = (16, 3, 67, 2);
        let (bytes, values) = q8_fixture(in_ch * kernel, out_ch);
        let weight = Weight::from_quantized(QuantizedTensor::from_bytes(
            &bytes,
            GGMLType::Q8_0,
            in_ch * kernel,
            out_ch,
        ));
        // Each activation block is exactly representable in Q8_0 (scale 1).
        let input: Vec<f32> = (0..in_ch * length)
            .map(|i| ((i * 7 % 3) as f32 - 1.0) * 127.0)
            .collect();
        let bias = [0.125, -0.25, 0.5];
        for (stride, dilation, pad) in [(1, 1, 1), (2, 1, 1), (1, 3, 3)] {
            let out_len = if stride == 1 {
                length
            } else {
                length.div_ceil(stride)
            };
            let actual = if stride == 1 {
                conv1d_causal(
                    &weight, &bias, &input, in_ch, length, out_ch, kernel, dilation, pad,
                )
            } else {
                conv1d_causal_strided(
                    &weight, &bias, &input, in_ch, length, out_ch, kernel, stride, pad,
                )
            };
            let mut expected = vec![0.0; out_ch * out_len];
            for oc in 0..out_ch {
                for t in 0..out_len {
                    let mut sum = 0.0;
                    for ic in 0..in_ch {
                        for k in 0..kernel {
                            let src = (t * stride + k * dilation) as isize - pad as isize;
                            if (0..length as isize).contains(&src) {
                                sum += values[(oc * in_ch + ic) * kernel + k]
                                    * input[ic * length + src as usize];
                            }
                        }
                    }
                    expected[oc * out_len + t] = sum + bias[oc];
                }
            }
            assert_eq!(actual, expected, "stride={stride}, dilation={dilation}");
            let float = f32_weight(values.clone(), in_ch * kernel);
            assert_eq!(
                conv1d(
                    &float, &bias, &input, in_ch, length, out_ch, kernel, stride, dilation, pad,
                    out_len
                ),
                expected,
            );
        }
    }

    #[test]
    fn q8_transpose_convolution_matches_scalar_scatter_and_causal_trim() {
        use crate::core::tensor::GGMLType;
        use crate::ops::kernel::QuantizedTensor;
        let (in_ch, out_ch, kernel, length, stride) = (3, 8, 4, 5, 2);
        let (bytes, values) = q8_fixture(out_ch * kernel, in_ch);
        let weight = Weight::from_quantized(QuantizedTensor::from_bytes(
            &bytes,
            GGMLType::Q8_0,
            out_ch * kernel,
            in_ch,
        ));
        let input: Vec<f32> = (0..in_ch * length)
            .map(|i| (i as f32 - 7.0) / 4.0)
            .collect();
        let bias: Vec<f32> = (0..out_ch).map(|c| c as f32 / 16.0).collect();
        let actual = conv_transpose1d_causal(
            &weight, &bias, &input, in_ch, length, out_ch, kernel, stride,
        );
        let raw_len = (length - 1) * stride + kernel;
        let mut raw = vec![0.0; out_ch * raw_len];
        for oc in 0..out_ch {
            for t in 0..raw_len {
                for ic in 0..in_ch {
                    for i in 0..length {
                        if t >= i * stride && t - i * stride < kernel {
                            raw[oc * raw_len + t] += input[ic * length + i]
                                * values[(ic * out_ch + oc) * kernel + t - i * stride];
                        }
                    }
                }
            }
        }
        let expected: Vec<f32> = (0..out_ch)
            .flat_map(|oc| {
                raw[oc * raw_len..oc * raw_len + length * stride]
                    .iter()
                    .map(|x| x + bias[oc])
                    .collect::<Vec<_>>()
            })
            .collect();
        assert_eq!(actual, expected);
        // Float formats also expand only the requested input-channel row.
        // The fixture values are exactly representable in all three.
        for dtype in [GGMLType::F32, GGMLType::F16, GGMLType::BF16] {
            let bytes: Vec<u8> = values
                .iter()
                .flat_map(|&value| match dtype {
                    GGMLType::F32 => value.to_le_bytes().to_vec(),
                    GGMLType::F16 => crate::ops::f32_to_f16(value).to_le_bytes().to_vec(),
                    GGMLType::BF16 => ((value.to_bits() >> 16) as u16).to_le_bytes().to_vec(),
                    _ => unreachable!(),
                })
                .collect();
            let mut float = Weight::from_quantized(QuantizedTensor::from_bytes(
                &bytes,
                dtype,
                out_ch * kernel,
                in_ch,
            ));
            float.n_in = out_ch * kernel;
            float.n_out = in_ch;
            assert_eq!(
                conv_transpose1d_causal(
                    &float, &bias, &input, in_ch, length, out_ch, kernel, stride
                ),
                expected,
                "{dtype:?}",
            );
        }
        assert!(
            conv_transpose1d_causal(&weight, &bias, &[], in_ch, 0, out_ch, kernel, stride)
                .is_empty()
        );
    }

    fn scalar_q8_activation(input: &[f32]) -> Vec<f32> {
        input
            .chunks_exact(32)
            .flat_map(|block| {
                let max = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                let scale = max / 127.0;
                let inverse = if scale == 0.0 { 0.0 } else { 1.0 / scale };
                let stored_scale = crate::ops::f16_to_f32(crate::ops::f32_to_f16(scale));
                block
                    .iter()
                    .map(move |&v| (v * inverse).round_ties_even() * stored_scale)
            })
            .collect()
    }

    #[test]
    fn q8_lstm_matches_independent_gate_order_bias_and_recurrent_state() {
        use crate::core::tensor::GGMLType;
        use crate::ops::kernel::QuantizedTensor;
        let mut bytes = Vec::new();
        // Sparse diagonal gates make a direct reference possible without
        // sharing the matrix kernel or materializing dense float matrices.
        let coefficients = [1i8, -2, 3, 2];
        for gate in 0..4 {
            for channel in 0..512 {
                for block in 0..16 {
                    bytes.extend_from_slice(&crate::ops::f32_to_f16(1.0 / 32.0).to_le_bytes());
                    for lane in 0..32 {
                        bytes.push(if block * 32 + lane == channel {
                            coefficients[gate] as u8
                        } else {
                            0
                        });
                    }
                }
            }
        }
        let weight = Weight::from_quantized(QuantizedTensor::from_bytes(
            &bytes,
            GGMLType::Q8_0,
            512,
            2048,
        ));
        let frames = 3;
        let input: Vec<f32> = (0..frames * 512)
            .map(|i| ((i * 11 % 37) as f32 - 18.0) / 23.0)
            .collect();
        let input_bias: Vec<f32> = (0..2048).map(|i| ((i % 7) as f32 - 3.0) / 16.0).collect();
        let hidden_bias: Vec<f32> = (0..2048).map(|i| ((i % 11) as f32 - 5.0) / 32.0).collect();
        let actual =
            lstm_layer_forward(&weight, &weight, &input_bias, &hidden_bias, &input, frames);
        let mut hidden = vec![0.0; 512];
        let mut cell = vec![0.0; 512];
        let mut expected = Vec::new();
        for frame in 0..frames {
            let xq = scalar_q8_activation(&input[frame * 512..(frame + 1) * 512]);
            let hq = scalar_q8_activation(&hidden);
            for channel in 0..512 {
                let gates: [f32; 4] = std::array::from_fn(|gate| {
                    let coefficient = coefficients[gate] as f32 / 32.0;
                    coefficient * hq[channel]
                        + hidden_bias[gate * 512 + channel]
                        + (coefficient * xq[channel] + input_bias[gate * 512 + channel])
                });
                let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
                cell[channel] =
                    sigmoid(gates[1]) * cell[channel] + sigmoid(gates[0]) * gates[2].tanh();
                hidden[channel] = sigmoid(gates[3]) * cell[channel].tanh();
            }
            expected.extend_from_slice(&hidden);
        }
        for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-6,
                "LSTM output[{index}]: {actual} vs {expected}"
            );
        }
    }
}
