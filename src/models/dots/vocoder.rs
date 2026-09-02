//! AudioVAE vocoder: prompt-latent extraction encoder (`audio_encoder` +
//! `enc_mi` + `pre_proj`) and the BigVGAN decoder (`post_proj` + `dec_mi` +
//! `decoder`) that turns 128-dim latent frames into 48 kHz mono audio.
//!
//! Weight-norm was folded into plain `.weight` at export time; the fixed
//! kaiser filters of the AMP-block activations were emitted as
//! `...activations.{a}.{up,down}_filter`; the post activation keeps its
//! trained filters.

use crate::core::tensor::TensorSource;
use crate::models::dots::patch_encoder::load_f16_f32;

const LEAKY: f32 = 0.2;
const SNAKE_EPS: f32 = 1e-9;
const HOP: usize = 1920; // product of decoder upsample rates

// ---------------------------------------------------------------------------
// small conv primitives (torch layout: conv1d [out,in,k])
// ---------------------------------------------------------------------------

fn conv1d_causal(
    weight: &[f32],
    bias: &[f32],
    input: &[f32],
    in_ch: usize,
    length: usize,
    out_ch: usize,
    kernel: usize,
    dilation: usize,
    left_pad: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; out_ch * length];
    for oc in 0..out_ch {
        for o in 0..length {
            let mut acc = bias[oc];
            for ic in 0..in_ch {
                for k in 0..kernel {
                    let src = o as isize * 1 + k as isize * dilation as isize - left_pad as isize;
                    if src >= 0 && src < length as isize {
                        let w = weight[oc * in_ch * kernel + ic * kernel + k];
                        acc += w * input[ic * length + src as usize];
                    }
                }
            }
            out[oc * length + o] = acc;
        }
    }
    out
}

/// Causal Conv1d with stride: output length = (length + left_pad - kernel)/stride + 1.
fn conv1d_causal_strided(
    weight: &[f32],
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
    let mut out = vec![0.0f32; out_ch * out_len];
    for oc in 0..out_ch {
        for o in 0..out_len {
            let mut acc = bias[oc];
            for ic in 0..in_ch {
                for k in 0..kernel {
                    let src = o as isize * stride as isize + k as isize - left_pad as isize;
                    if src >= 0 && src < length as isize {
                        let w = weight[oc * in_ch * kernel + ic * kernel + k];
                        acc += w * input[ic * length + src as usize];
                    }
                }
            }
            out[oc * out_len + o] = acc;
        }
    }
    out
}

fn conv1d_pad2(
    weight: &[f32],
    bias: &[f32],
    input: &[f32],
    in_ch: usize,
    length: usize,
    out_ch: usize,
    kernel: usize,
    pad: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; out_ch * length];
    for oc in 0..out_ch {
        for o in 0..length {
            let mut acc = bias[oc];
            for ic in 0..in_ch {
                for k in 0..kernel {
                    let src = o as isize + k as isize - pad as isize;
                    if src >= 0 && src < length as isize {
                        let w = weight[oc * in_ch * kernel + ic * kernel + k];
                        acc += w * input[ic * length + src as usize];
                    }
                }
            }
            out[oc * length + o] = acc;
        }
    }
    out
}

/// Causal ConvTranspose1d (kernel == 2*stride, pad 0): output length = in*stride.
fn conv_transpose1d_causal(
    weight: &[f32],
    bias: &[f32],
    input: &[f32],
    in_ch: usize,
    length: usize,
    out_ch: usize,
    kernel: usize,
    stride: usize,
) -> Vec<f32> {
    // raw output is (length-1)*stride + kernel long; causal trim drops the
    // last `stride` samples so the result is exactly length*stride
    let raw_len = (length - 1) * stride + kernel;
    let out_len = length * stride;
    let mut raw = vec![0.0f32; out_ch * raw_len];
    for ic in 0..in_ch {
        for oc in 0..out_ch {
            for i in 0..length {
                let x = input[ic * length + i];
                if x == 0.0 {
                    continue;
                }
                for k in 0..kernel {
                    let w = weight[ic * out_ch * kernel + oc * kernel + k];
                    raw[oc * raw_len + i * stride + k] += w * x;
                }
            }
        }
    }
    for oc in 0..out_ch {
        for n in 0..out_len {
            raw[oc * raw_len + n] += bias[oc];
        }
    }
    // causal trim
    let mut out = Vec::with_capacity(out_ch * out_len);
    for oc in 0..out_ch {
        out.extend_from_slice(&raw[oc * raw_len..oc * raw_len + out_len]);
    }
    out
}

fn leaky_inplace(x: &mut [f32]) {
    for value in x.iter_mut() {
        *value = if *value > 0.0 { *value } else { LEAKY * *value };
    }
}

/// SnakeBeta with logscale parameters.
fn snakebeta(x: f32, alpha: f32, beta: f32) -> f32 {
    let (a, b) = (alpha.exp(), beta.exp());
    x + (x * a).sin().powi(2) / (b + SNAKE_EPS)
}

// ---------------------------------------------------------------------------
// MI layers (Linear + skip-LSTM + Linear)
// ---------------------------------------------------------------------------

pub struct MiLayer {
    pub lin0_w: Vec<f32>,
    pub lin0_b: Vec<f32>,
    /// l = 0..4: (weight_ih [2048,512], weight_hh, bias_ih, bias_hh)
    pub lstm: [(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>); 4],
    pub lin2_w: Vec<f32>,
    pub lin2_b: Vec<f32>,
}

impl MiLayer {
    fn from_source(source: &dyn TensorSource, prefix: &str) -> Result<Self, String> {
        let mut lstm = Vec::with_capacity(4);
        for l in 0..4 {
            lstm.push((
                load_f16_f32(source, &format!("{prefix}.1.lstm.weight_ih_l{l}"), &[512, 2048])?,
                load_f16_f32(source, &format!("{prefix}.1.lstm.weight_hh_l{l}"), &[512, 2048])?,
                load_f16_f32(source, &format!("{prefix}.1.lstm.bias_ih_l{l}"), &[2048])?,
                load_f16_f32(source, &format!("{prefix}.1.lstm.bias_hh_l{l}"), &[2048])?,
            ));
        }
        Ok(Self {
            lin0_w: load_f16_f32(source, &format!("{prefix}.0.weight"), &[128, 512])?,
            lin0_b: load_f16_f32(source, &format!("{prefix}.0.bias"), &[512])?,
            lstm: lstm.try_into().unwrap(),
            lin2_w: load_f16_f32(source, &format!("{prefix}.2.weight"), &[512, 128])?,
            lin2_b: load_f16_f32(source, &format!("{prefix}.2.bias"), &[128])?,
        })
    }

    /// `x` is `[frames, 128]`; returns `[frames, 128]`.
    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let frames = x.len() / 128;
        let mut h = vec![0.0f32; frames * 512];
        for f in 0..frames {
            for o in 0..512 {
                let mut acc = self.lin0_b[o];
                for i in 0..128 {
                    acc += self.lin0_w[o * 128 + i] * x[f * 128 + i];
                }
                h[f * 512 + o] = acc;
            }
        }
        // LSTM: 4 layers, gate order i/f/g/o
        let mut layer_in = h.clone();
        for layer in 0..4 {
            let (w_ih, w_hh, b_ih, b_hh) = &self.lstm[layer];
            // w dims [512, 2048] → per output gating row: w[i*2048 + g*512 + o]
            let mut out = vec![0.0f32; frames * 512];
            let mut c = vec![0.0f32; 512];
            let mut hx = vec![0.0f32; 512];
            for f in 0..frames {
                let mut gates = [0.0f32; 2048];
                for g in 0..2048 {
                    let mut acc = b_ih[g] + b_hh[g];
                    for i in 0..512 {
                        // weight layout [512, 2048] (gguf dims fastest first):
                        // element (i, g) sits at i + g * 512
                        acc += w_ih[i + g * 512] * layer_in[f * 512 + i];
                        acc += w_hh[i + g * 512] * hx[i];
                    }
                    gates[g] = acc;
                }
                for o in 0..512 {
                    let i_g = sigmoid(gates[o]);
                    let f_g = sigmoid(gates[512 + o]);
                    let g_g = gates[1024 + o].tanh();
                    let o_g = sigmoid(gates[1536 + o]);
                    c[o] = f_g * c[o] + i_g * g_g;
                    hx[o] = o_g * c[o].tanh();
                    out[f * 512 + o] = hx[o];
                }
            }
            layer_in = out;
        }
        if !self.lstm.is_empty() {
            // skip connection: h + residual
            for (a, b) in layer_in.iter_mut().zip(h.iter()) {
                *a += b;
            }
        }
        let mut y = vec![0.0f32; frames * 128];
        for f in 0..frames {
            for o in 0..128 {
                let mut acc = self.lin2_b[o];
                for i in 0..512 {
                    acc += self.lin2_w[o * 512 + i] * layer_in[f * 512 + i];
                }
                y[f * 128 + o] = acc;
            }
        }
        y
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// ---------------------------------------------------------------------------
// AudioVAE encoder (prompt latent extraction; causal)
// ---------------------------------------------------------------------------

struct EncConv {
    weight: Vec<f32>,
    bias: Vec<f32>,
    kernel: usize,
    stride: usize,
    out_ch: usize,
}

struct EncResStackLayer {
    c1: Vec<f32>,
    b1: Vec<f32>,
    d1: usize,
    c2: Vec<f32>,
    b2: Vec<f32>,
}

struct EncResStack {
    layers: Vec<EncResStackLayer>,
    ch: usize,
}

pub struct AudioEncoder {
    convs: Vec<EncConv>,        // 8 convs: [pre(1→12), 6 down, post(768→128)]
    resstacks: Vec<EncResStack>, // 6
}

impl AudioEncoder {
    fn from_source(source: &dyn TensorSource) -> Result<Self, String> {
        let w = |name: &str, dims: &[u64]| -> Result<Vec<f32>, String> { load_f16_f32(source, name, dims) };
        let conv = |idx: usize, in_ch: usize, out_ch: usize, kernel: usize, stride: usize| -> Result<EncConv, String> {
            Ok(EncConv {
                weight: w(
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
                    c1: w(
                        &format!("dotstts.vocoder.audio_encoder.generator.{gi}.layers.{j}.2.weight"),
                        &[3, ch as u64, ch as u64],
                    )?,
                    b1: w(
                        &format!("dotstts.vocoder.audio_encoder.generator.{gi}.layers.{j}.2.bias"),
                        &[ch as u64],
                    )?,
                    d1: d,
                    c2: w(
                        &format!("dotstts.vocoder.audio_encoder.generator.{gi}.layers.{j}.5.weight"),
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
            let in_ch = if ci == 0 { 1 } else { self.convs[ci - 1].out_ch };
            let left_pad = conv.kernel - 1; // causal, dilation 1
            out = conv1d_causal_strided(
                &conv.weight,
                &conv.bias,
                &out,
                in_ch,
                length,
                conv.out_ch,
                conv.kernel,
                conv.stride,
                left_pad,
            );
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
            // conv1 d=layer.d1 causal pad d*(k-1)=2*d
            let pad1 = 2 * layer.d1;
            h = conv1d_causal(&layer.c1, &layer.b1, &h, rs.ch, length, rs.ch, 3, layer.d1, pad1);
            for value in h.iter_mut() {
                *value = if *value > 0.0 { *value } else { LEAKY * *value };
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

struct AmpConv {
    weight: Vec<f32>,
    bias: Vec<f32>,
    kernel: usize,
    dilation: usize,
}

pub(crate) struct AmpBlock {
    convs1: Vec<AmpConv>,
    convs2: Vec<AmpConv>,
    alphas: Vec<f32>, // 6 [ch]
    betas: Vec<f32>,  // 6 [ch]
    ch: usize,
    up_filter: Vec<f32>,   // fixed kaiser [12]
    down_filter: Vec<f32>, // fixed kaiser [12]
}

impl AmpBlock {
    fn from_source(
        source: &dyn TensorSource,
        idx: usize,
        ch: usize,
        kernel: usize,
    ) -> Result<Self, String> {
        let w = |name: &str, dims: &[u64]| -> Result<Vec<f32>, String> { load_f16_f32(source, name, dims) };
        let conv = |group: &str, j: usize, dilation: usize| -> Result<AmpConv, String> {
            Ok(AmpConv {
                weight: w(
                    &format!(
                        "dotstts.vocoder.decoder.resblocks.{idx}.{group}.{j}.weight"
                    ),
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
            &format!("dotstts.vocoder.decoder.resblocks.{idx}.activations.0.up_filter"),
            &[12],
        )?;
        let down_filter = w(
            &format!("dotstts.vocoder.decoder.resblocks.{idx}.activations.0.down_filter"),
            &[12],
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

    fn forward(&self, x: &[f32], length: usize) -> Vec<f32> {
        let mut cur = x.to_vec();
        for j in 0..3 {
            // act a[2j] → conv1 → act a[2j+1] → conv2 → residual
            cur = self.act_j(cur.clone(), length, 2 * j);
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
            cur = self.act_j(cur, length, 2 * j + 1);
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
            for (a, b) in cur.iter_mut().zip(x.iter()) {
                *a += b;
            }
        }
        cur
    }

    fn act_j(&self, x: Vec<f32>, length: usize, a: usize) -> Vec<f32> {
        // Activation1d: upsample (fixed filter) → snakebeta → downsample
        let up_len = 2 * length + 11;
        let mut up = vec![0.0f32; self.ch * up_len];
        for c in 0..self.ch {
            for i in 0..length {
                let val = x[c * length + i];
                if val == 0.0 {
                    continue;
                }
                for k in 0..12usize {
                    // reference multiplies the transposed conv by ratio=2
                    up[c * up_len + i * 2 + k] += 2.0 * self.up_filter[k] * val;
                }
            }
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
        down
    }
}

pub struct BigVganDecoder {
    pub conv_pre: (Vec<f32>, Vec<f32>), // [5,128,1536]
    pub ups: Vec<(Vec<f32>, Vec<f32>, usize, usize, usize, usize)>, // w,b,kernel,stride,in,out
    pub(crate) resblocks: Vec<AmpBlock>,
    pub post_alpha: Vec<f32>,
    pub post_beta: Vec<f32>,
    pub post_up: Vec<f32>,   // trained [24,1,12] → flattened per-channel [24*12]
    pub post_down: Vec<f32>, // trained [24,1,12]
    pub conv_post: (Vec<f32>, Vec<f32>), // [7,24,1]
}

impl BigVganDecoder {
    fn from_source(source: &dyn TensorSource) -> Result<Self, String> {
        let w = |name: &str, dims: &[u64]| -> Result<Vec<f32>, String> { load_f16_f32(source, name, dims) };
        let conv_pre = (
            w("dotstts.vocoder.decoder.conv_pre.weight", &[5, 128, 1536])?,
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
                w(
                    &format!("dotstts.vocoder.decoder.ups.{i}.0.weight"),
                    &[k as u64, och as u64, ich as u64],
                )?,
                w(&format!("dotstts.vocoder.decoder.ups.{i}.0.bias"), &[och as u64])?,
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
                resblocks.push(AmpBlock::from_source(
                    source,
                    idx,
                    ch,
                    stage_kernels[j],
                )?);
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
            w("dotstts.vocoder.decoder.conv_post.weight", &[7, 24, 1])?,
            // the checkpoint ships no conv_post bias; keep zeros so the
            // decoder math is unchanged
            w("dotstts.vocoder.decoder.conv_post.bias", &[1]).unwrap_or_else(|_| vec![0.0]),
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
        for i in 0..6 {
            let (w_, b, kernel, stride, in_ch, out_ch) = &self.ups[i];
            cur = conv_transpose1d_causal(
                w_,
                b,
                &cur,
                *in_ch,
                length,
                *out_ch,
                *kernel,
                *stride,
            );
            length *= stride;
            // 3 AMP blocks summed
            let mut acc = self.resblocks[i * 3].forward(&cur, length);
            let sum2 = self.resblocks[i * 3 + 1].forward(&cur, length);
            for (a, b) in acc.iter_mut().zip(sum2.iter()) {
                *a += b;
            }
            let sum3 = self.resblocks[i * 3 + 2].forward(&cur, length);
            for (a, b) in acc.iter_mut().zip(sum3.iter()) {
                *a += b;
            }
            for value in acc.iter_mut() {
                *value /= 3.0;
            }
            cur = acc;
        }
        // activation_post (trained filters) + conv_post + clamp
        let ch = 24usize;
        let up_len = 2 * length + 11;
        let mut up = vec![0.0f32; ch * up_len];
        for c in 0..ch {
            for i in 0..length {
                let val = cur[c * length + i];
                if val == 0.0 {
                    continue;
                }
                for k in 0..12usize {
                    // reference multiplies the transposed conv by ratio=2
                    up[c * up_len + i * 2 + k] += 2.0 * self.post_up[c * 12 + k] * val;
                }
            }
        }
        let mut snake = vec![0.0f32; ch * 2 * length];
        for c in 0..ch {
            let (alpha, beta) = (self.post_alpha[c], self.post_beta[c]);
            for n in 0..2 * length {
                snake[c * 2 * length + n] = snakebeta(up[c * up_len + n], alpha, beta);
            }
        }
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
        // conv_post: [24, L] → [1, L] causal pad 6
        let final_len = length;
        let mut audio = vec![0.0f32; final_len];
        for n in 0..final_len {
            let mut acc = self.conv_post.1[0];
            for ic in 0..24 {
                for k in 0..7usize {
                    let src = n as isize + k as isize - 6;
                    if src >= 0 && src < final_len as isize {
                        acc += self.conv_post.0[ic * 7 + k] * down[ic * final_len + src as usize];
                    }
                }
            }
            audio[n] = acc.clamp(-1.0, 1.0); // use_tanh_at_final=False → clamp
        }
        audio
    }
}

// ---------------------------------------------------------------------------
// Assembled vocoder
// ---------------------------------------------------------------------------

pub struct Vocoder {
    pub(crate) encoder: AudioEncoder,
    pub(crate) enc_mi: MiLayer,
    pub pre_proj: (Vec<f32>, Vec<f32>), // [1,128,256]
    pub post_proj: (Vec<f32>, Vec<f32>), // [1,128,128]
    pub(crate) dec_mi: MiLayer,
    pub(crate) decoder: BigVganDecoder,
}

impl Vocoder {
    pub fn from_source(source: &dyn TensorSource) -> Result<Self, String> {
        let w = |name: &str, dims: &[u64]| -> Result<Vec<f32>, String> { load_f16_f32(source, name, dims) };
        Ok(Self {
            encoder: AudioEncoder::from_source(source)?,
            enc_mi: MiLayer::from_source(source, "dotstts.vocoder.enc_mi_layer")?,
            pre_proj: (
                w("dotstts.vocoder.pre_proj.weight", &[1, 128, 256])?,
                w("dotstts.vocoder.pre_proj.bias", &[256])?,
            ),
            post_proj: (
                w("dotstts.vocoder.post_proj.weight", &[1, 128, 128])?,
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
        // permute to [T, 128], MI layers
        let mut frames_in = vec![0.0f32; frames * 128];
        for c in 0..128 {
            for t in 0..frames {
                frames_in[t * 128 + c] = encoded[c * frames + t];
            }
        }
        let mi_out = self.enc_mi.forward(&frames_in); // [T, 128]
        // permute back to [128, T], then pre_proj 1x1 conv (128 → 256)
        let mut out = vec![0.0f32; 256 * frames];
        for o in 0..256 {
            for t in 0..frames {
                let mut acc = self.pre_proj.1[o];
                for c in 0..128 {
                    // 1x1 conv weight [out=256, in=128] → offset c + o*128
                    acc += self.pre_proj.0[c + o * 128] * mi_out[t * 128 + c];
                }
                out[o * frames + t] = acc;
            }
        }
        Ok(out)
    }

    /// Decode `[frames, 128]` raw latents → 48 kHz mono waveform.
    pub fn decode_latents(&self, latents: &[f32]) -> Result<Vec<f32>, String> {
        let frames = latents.len() / 128;
        if latents.len() != frames * 128 || frames == 0 {
            return Err("vocoder decode_latents expects [frames, 128]".into());
        }
        // post_proj (1x1, 128→128)
        let mut post = vec![0.0f32; 128 * frames];
        for o in 0..128 {
            for t in 0..frames {
                let mut acc = self.post_proj.1[o];
                for c in 0..128 {
                    // 1x1 conv weight [out=128, in=128] → offset c + o*128
                    acc += self.post_proj.0[c + o * 128] * latents[t * 128 + c];
                }
                post[o * frames + t] = acc;
            }
        }
        // permute → [T, 128] → dec_mi → permute → decoder
        let mut mi_in = vec![0.0f32; frames * 128];
        for c in 0..128 {
            for t in 0..frames {
                mi_in[t * 128 + c] = post[c * frames + t];
            }
        }
        let mi_out = self.dec_mi.forward(&mi_in); // [T, 128]
        let mut dec_in = vec![0.0f32; 128 * frames];
        for t in 0..frames {
            for c in 0..128 {
                dec_in[c * frames + t] = mi_out[t * 128 + c];
            }
        }
        Ok(self.decoder.forward(&dec_in))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snakebeta_matches_reference_formula() {
        // y = x + sin²(x·exp(a)) / (exp(b) + 1e-9)
        let y = snakebeta(0.5, 0.0, 0.0);
        let expected = 0.5 + (0.5f32).sin().powi(2) / (1.0 + SNAKE_EPS);
        assert!((y - expected).abs() < 1e-6);
    }

#[test]
    fn conv_transpose_causal_produces_exact_length() {
        let w = vec![1.0f32; 4 * 2 * 4]; // in=4,out=2,k=4
        let b = vec![0.0f32; 2];
        let x = vec![1.0f32; 4 * 3]; // 3 frames
        let y = conv_transpose1d_causal(&w, &b, &x, 4, 3, 2, 4, 2);
        assert_eq!(y.len(), 2 * 3 * 2);
    }
}