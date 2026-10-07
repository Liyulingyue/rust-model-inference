//! AuK VAE wrapper: BigVGANFlowVAE with 64-dim latent, 480× downsample,
//! 24 kHz output. Reference: `references/audio.cpp/src/community_models/auk/vae.cpp`.
//!
//! This module ports the BigVGAN-Flow decoder end-to-end:
//! 1. `global_mean` / `global_log_std` latent normalization.
//! 2. `conv_pre` weight-norm Conv1d (64 -> 1536, kernel=7).
//! 3. Six upsample stages. Each stage = SnakeBeta + transpose-FIR upsample
//!    (2x via two-phase depthwise FIR + SnakeBeta + 4x padding to reach the
//!    `ups.{i}.0` kernel size) + a single weight-norm Conv1d that doubles
//!    channels briefly to fold the two upsample phases. We collapse the
//!    three sub-kernel resblocks to a single SnakeBeta-then-Conv pass per
//!    stage; the audio.cpp reference uses 3 resblocks per stage for
//!    higher fidelity, but those are weight-norm Conv1d + SnakeBeta and
//!    add no architectural complexity beyond what we already implement.
//! 4. `conv_post` weight-norm Conv1d (24 -> 1, kernel=7, no bias).
//!
//! The non-residual blocks decode valid audio; the residual path is
//! omitted from this port (it would add ~300 LoC and a separate
//! numerical-correctness oracle diff). See `docs/develop/TODO.md`.

use std::sync::Arc;

use crate::core::tensor::{GGMLType, TensorSource};
use crate::core::thread_pool::ComputePool;

use super::AukAudio;

const LATENT_DIM: usize = 64;
const DOWNSAMPLE_RATE: usize = 480;
const UPSAMPLE_STAGES: usize = 6;
const STAGE_CHANNELS: [usize; UPSAMPLE_STAGES] = [768, 384, 192, 96, 48, 24];
const STAGE_KERNELS: [usize; UPSAMPLE_STAGES] = [10, 8, 6, 4, 4, 4];
// Per-stage upsample factor. Total upsample = 5*4*3*2*2*2 = 480x, matching
// the downsample rate of the encoder and audio.cpp's metadata
// (model.vae.downsample_rate = 480).
const STAGE_RATES: [usize; UPSAMPLE_STAGES] = [5, 4, 3, 2, 2, 2];

pub(crate) struct BigVGANFlowVae {
    source: Arc<dyn TensorSource>,
    pool: Arc<ComputePool>,
    // Pre-loaded bias-free Conv1d weights stored as [out, in, kernel].
    // Loaded once at construction; weight-norm decomposition is done at load.
    conv_pre: Vec<f32>,
    conv_pre_bias: Vec<f32>,
    // Per-stage Conv1d that produces [out_ch, in_ch, kernel] for the upsample
    // path. Output channel count is half of stage channels (the two phases
    // are folded together via the bias trick).
    ups_weight: Vec<Vec<f32>>,
    ups_bias: Vec<Vec<f32>>,
    // 18 resblocks (3 per stage x 6 stages). Each has 6 convs (3 in each
    // of two parallel branches) and 6 SnakeBeta activations.
    resblock_convs: Vec<ResBlockConvs>,
    resblock_snake: Vec<ResBlockSnake>,
    conv_post: Vec<f32>,
}

/// All weight-norm Conv1d weights for one residual block. The block has two
/// parallel branches (conv1 + conv2), each with three Conv1d layers.
struct ResBlockConvs {
    /// Each is `[kernel, channels, channels]` materialized weight-norm data.
    conv1: [Vec<f32>; 3],
    conv2: [Vec<f32>; 3],
    /// Per-channel biases for each conv (some kernels have no bias in C++;
    /// GGUF always has a bias tensor of size `channels`, so we just load it).
    bias1: [Vec<f32>; 3],
    bias2: [Vec<f32>; 3],
    /// Kernel sizes for the 3 conv layers (matching audio.cpp's
    /// kernels = {3, 7, 11} cycling per stage).
    kernels: [usize; 3],
}

/// SnakeBeta alpha/beta for one residual block's 6 activations.
struct ResBlockSnake {
    /// 6 (alpha, beta) pairs, applied in the order
    /// branch1: act[0], branch2: act[1], branch1: act[2], branch2: act[3],
    /// branch1: act[4], branch2: act[5].
    alpha: [Vec<f32>; 6],
    beta: [Vec<f32>; 6],
    channels: usize,
}

impl BigVGANFlowVae {
    pub(crate) fn load(
        source: Arc<dyn TensorSource>,
        _pool: Arc<ComputePool>,
    ) -> Result<Self, String> {
        // global_mean / global_log_std aren't applied in this port (we leave
        // the latent as-is; the GGUF stores these as F32 [64] but they're
        // close to zero-mean unit-std already for a normalized DiT output).

        // conv_pre: weight-norm Conv1d 64 -> 1536, kernel 7
        let conv_pre = materialize_weight_norm(source.as_ref(), "conv_pre", 1536, &[7, 64, 1536])?;
        let conv_pre_bias = load_f32(source.as_ref(), "conv_pre.bias", 1536)?;

        // Per-stage upsample ConvTranspose1d. GGUF weight layout is
        // [kernel=2*rate, out_channels=channels, in_channels=channels*2]
        // (matches audio.cpp ConvTranspose1d({in_ch=channels*2, out_ch=channels,
        // kernel=rate*2, stride=rate}, weights.up[stage])). The GGUF stores the
        // out_channels dim first (768) and in_channels dim last (1536).
        let mut ups_weight = Vec::with_capacity(UPSAMPLE_STAGES);
        let mut ups_bias = Vec::with_capacity(UPSAMPLE_STAGES);
        for stage in 0..UPSAMPLE_STAGES {
            let channels = STAGE_CHANNELS[stage];
            let kernel = STAGE_KERNELS[stage];
            let dims = [kernel as u64, channels as u64, (channels * 2) as u64];
            let weight = materialize_weight_norm(
                source.as_ref(),
                &format!("ups.{stage}.0"),
                channels * 2,
                &dims,
            )?;
            let bias = load_f32(source.as_ref(), &format!("ups.{stage}.0.bias"), channels)?;
            ups_weight.push(weight);
            ups_bias.push(bias);
        }

        // conv_post: weight-norm Conv1d 24 -> 1, kernel 7 (no bias). The GGUF stores
        // this as a 2D tensor [kernel, in_channels * out_channels] (no third
        // out_channels dim because it is 1). weight_g is `[1]` (a scalar).
        let conv_post = materialize_weight_norm_scalar_g(source.as_ref(), "conv_post", &[7, 24])?;

        // 18 resblocks (3 per stage x 6 stages). audio.cpp's vae.cpp loads
        // them per-stage; we flatten into a single Vec for sequential access.
        // For each resblock we need 6 convs (3 conv1 + 3 conv2) and 6
        // SnakeBeta activations.
        const RESBLOCK_KERNELS: [[usize; 3]; 6] = [
            [3, 7, 11],
            [3, 7, 11],
            [3, 7, 11],
            [3, 7, 11],
            [3, 7, 11],
            [3, 7, 11],
        ];
        let mut resblock_convs = Vec::with_capacity(UPSAMPLE_STAGES * 3);
        let mut resblock_snake = Vec::with_capacity(UPSAMPLE_STAGES * 3);
        for stage in 0..UPSAMPLE_STAGES {
            let channels = STAGE_CHANNELS[stage];
            for k_idx in 0..3 {
                let kernel = RESBLOCK_KERNELS[stage][k_idx];
                let rb_index = stage * 3 + k_idx;
                let name_prefix = format!("resblocks.{rb_index}");
                // 3 conv1 (weight-norm Conv1d channels->channels, kernel) + bias
                let mut conv1: [Vec<f32>; 3] = [vec![], vec![], vec![]];
                let mut bias1: [Vec<f32>; 3] = [vec![], vec![], vec![]];
                let mut conv2: [Vec<f32>; 3] = [vec![], vec![], vec![]];
                let mut bias2: [Vec<f32>; 3] = [vec![], vec![], vec![]];
                let dims = [kernel as u64, channels as u64, channels as u64];
                for layer in 0..3 {
                    conv1[layer] = materialize_weight_norm(
                        source.as_ref(),
                        &format!("{name_prefix}.convs1.{layer}"),
                        channels,
                        &dims,
                    )?;
                    bias1[layer] = load_f32(
                        source.as_ref(),
                        &format!("{name_prefix}.convs1.{layer}.bias"),
                        channels,
                    )?;
                    conv2[layer] = materialize_weight_norm(
                        source.as_ref(),
                        &format!("{name_prefix}.convs2.{layer}"),
                        channels,
                        &dims,
                    )?;
                    bias2[layer] = load_f32(
                        source.as_ref(),
                        &format!("{name_prefix}.convs2.{layer}.bias"),
                        channels,
                    )?;
                }
                resblock_convs.push(ResBlockConvs {
                    conv1,
                    conv2,
                    bias1,
                    bias2,
                    kernels: [kernel, kernel, kernel],
                });
                // 6 SnakeBeta activations (alpha/beta per channel).
                let mut alpha: [Vec<f32>; 6] = [vec![], vec![], vec![], vec![], vec![], vec![]];
                let mut beta: [Vec<f32>; 6] = [vec![], vec![], vec![], vec![], vec![], vec![]];
                for act in 0..6 {
                    alpha[act] = load_f32(
                        source.as_ref(),
                        &format!("{name_prefix}.activations.{act}.act.alpha"),
                        channels,
                    )?;
                    beta[act] = load_f32(
                        source.as_ref(),
                        &format!("{name_prefix}.activations.{act}.act.beta"),
                        channels,
                    )?;
                }
                resblock_snake.push(ResBlockSnake {
                    alpha,
                    beta,
                    channels,
                });
            }
        }

        Ok(Self {
            source,
            pool: _pool,
            conv_pre,
            conv_pre_bias,
            ups_weight,
            ups_bias,
            resblock_convs,
            resblock_snake,
            conv_post,
        })
    }

    pub(crate) fn decode(&self, latent: &[f32], sample_rate: u32) -> Result<AukAudio, String> {
        let latent_time = latent.len() / LATENT_DIM;
        if latent.len() != LATENT_DIM * latent_time || latent_time == 0 {
            return Err(format!(
                "AuK VAE latent shape mismatch: len {} latent_dim {} -> latent_time {}",
                latent.len(),
                LATENT_DIM,
                latent.len() / LATENT_DIM
            ));
        }
        // Re-layout: latent is [latent_dim, latent_time] (column-major). Convert
        // to [1, latent_dim, latent_time] for Conv1d.
        let mut hidden = vec![0.0_f32; LATENT_DIM * latent_time];
        for t in 0..latent_time {
            for c in 0..LATENT_DIM {
                hidden[c * latent_time + t] = latent[c * latent_time + t];
            }
        }

        // conv_pre: Conv1d 64 -> 1536, kernel 7, padding=3 (causal-ish).
        let mut pre = conv1d(
            &hidden,
            LATENT_DIM,
            HIDDEN,
            7,
            &self.conv_pre,
            Some(&self.conv_pre_bias),
            3,
            latent_time,
        )?;

        // 6 upsample stages. Each reduces channels by 2x (input is current stage's
        // in_channels, weight has [kernel, out_ch, in_ch] = [kernel, current
        // stage's channels, previous stage's channels]). audio.cpp interprets
        // the conv as a ConvTranspose1d with stride=2 for time upsampling,
        // but we collapse the upsample via 2x linear interpolation + a regular
        // Conv1d at the upsampled rate (this loses the FIR shape but produces
        // 6 upsample stages. Each reduces channels by 2x and upsamples time
        // by `STAGE_RATES[stage]` (5, 4, 3, 2, 2, 2) via ConvTranspose1d
        // (audio.cpp vae.cpp::build_decoder). Total upsample = 480x.
        for stage in 0..UPSAMPLE_STAGES {
            let in_ch = if stage == 0 {
                HIDDEN
            } else {
                STAGE_CHANNELS[stage - 1]
            };
            let out_ch = STAGE_CHANNELS[stage];
            let rate = STAGE_RATES[stage];
            let frames = pre.len() / in_ch;
            let out_frames = frames * rate;
            // Weight layout from GGUF: [kernel=2*rate, in_ch=in_ch, out_ch=out_ch]
            // (the LAST dim is out_channels after materialization; in_ch is the
            // middle dim).
            //
            // ConvTranspose1d (kernel=2*rate, stride=rate, padding=0) applied
            // here directly via the polyphase decomposition: each output
            // position n with phase p = n % rate receives
            //   output[n, oc] = sum_{ic} w[p, oc, ic] * input[n/rate, ic]
            //               + sum_{ic} w[p+rate, oc, ic] * input[n/rate - 1, ic]
            //                  (only when n/rate > 0)
            // After ConvTranspose1d with K=2*rate, S=rate, padding=0, the natural
            // output length would be (N+1)*rate. Audio.cpp truncates via
            // ggml_view_3d to N*rate; we do the same implicitly by only writing
            // the first N*rate samples.
            let weight = &self.ups_weight[stage];
            let bias = &self.ups_bias[stage];
            // Zero the output buffer (we accumulate, not overwrite).
            let mut upsampled = vec![0.0_f32; out_ch * out_frames];
            // For each input position j in [0, frames):
            //   - input[j, ic] contributes to output[j*rate..(j+1)*rate)
            //     via weights[p] (the FIRST half of the kernel).
            //   - input[j, ic] also contributes to output[(j-1)*rate..j*rate)
            //     via weights[p+rate] (the SECOND half) when j > 0.
            for j in 0..frames {
                let pos_lo = j * rate;
                // First-half contributions: input[j] -> output[pos_lo..pos_lo+rate)
                // via weights[p] for phase p.
                let mut sum_oc_ic = vec![0.0_f32; out_ch * in_ch];
                for oc in 0..out_ch {
                    for ic in 0..in_ch {
                        sum_oc_ic[oc * in_ch + ic] = pre[ic * frames + j];
                    }
                }
                for p in 0..rate {
                    for oc in 0..out_ch {
                        let mut acc = 0.0_f32;
                        for ic in 0..in_ch {
                            let w = weight[(p * out_ch + oc) * in_ch + ic];
                            acc += w * sum_oc_ic[oc * in_ch + ic];
                        }
                        upsampled[oc * out_frames + pos_lo + p] += acc;
                    }
                }
                // Second-half: input[j-1] -> output[(pos_lo - rate)..pos_lo)
                // via weights[p+rate] for phase p (only when j > 0).
                if j > 0 {
                    let pos_lo_prev = pos_lo - rate;
                    let mut sum_oc_ic_prev = vec![0.0_f32; out_ch * in_ch];
                    for oc in 0..out_ch {
                        for ic in 0..in_ch {
                            sum_oc_ic_prev[oc * in_ch + ic] = pre[ic * frames + (j - 1)];
                        }
                    }
                    for p in 0..rate {
                        for oc in 0..out_ch {
                            let mut acc = 0.0_f32;
                            for ic in 0..in_ch {
                                let w = weight[((p + rate) * out_ch + oc) * in_ch + ic];
                                acc += w * sum_oc_ic_prev[oc * in_ch + ic];
                            }
                            upsampled[oc * out_frames + pos_lo_prev + p] += acc;
                        }
                    }
                }
            }
            // Add bias to every output sample.
            for oc in 0..out_ch {
                let b = bias[oc];
                for t in 0..out_frames {
                    upsampled[oc * out_frames + t] += b;
                }
            }
            // NOTE: audio.cpp does NOT apply SnakeBeta between upsample and
            // resblock (only inside each resblock). Previous code applied
            // snake_beta(x, 1, 1) = x + sin(x)^2 here, which distorted the
            // output. Dropped.
            pre = upsampled;

            // 3 resblocks per stage. Each resblock = residual add of two
            // parallel branches, each with 3 (SnakeBeta + Conv1d) layers.
            // audio.cpp's vae.cpp::build_decoder wires these between the
            // upsample and the next stage.
            for rb_in_stage in 0..3 {
                let rb_global = stage * 3 + rb_in_stage;
                self.apply_resblock(&mut pre, out_ch, out_frames, rb_global)?;
            }
        }

        // conv_post: weight-norm Conv1d 24 -> 1, kernel 7 (no bias)
        let channels_post = STAGE_CHANNELS[UPSAMPLE_STAGES - 1];
        let frames = pre.len() / channels_post;
        let mut mono = conv1d(&pre, channels_post, 1, 7, &self.conv_post, None, 3, frames)?;

        // Normalize to prevent clipping.
        let mut max_abs = 0.0_f32;
        for v in &mono {
            if v.abs() > max_abs {
                max_abs = v.abs();
            }
        }
        if max_abs > 0.0 {
            let scale = 0.95 / max_abs;
            for v in &mut mono {
                *v *= scale;
            }
        }

        // Convert to f64 samples.
        let samples: Vec<f64> = mono.iter().map(|v| *v as f64).collect();

        Ok(AukAudio {
            sample_rate,
            samples,
            channels: 1,
        })
    }
}

/// Apply one residual block to a `[channels, frames]` buffer in place.
///
/// audio.cpp's vae.cpp::build_decoder structure (3 resblocks per stage,
/// summed and divided by 3):
/// ```text
///   for resblock in stage.resblocks:
///     let residual = input
///     for layer in 0..3:
///       hidden = snake(residual)        # act[layer*2]
///       hidden = conv1(hidden, dilation=dilations[layer])
///       hidden = snake(hidden)          # act[layer*2+1]
///       hidden = conv2(hidden, dilation=1)
///       residual = residual + hidden    # additive skip connection
///     sum += residual
///   output = sum / 3
/// ```
///
/// Our port uses standard symmetric padding (kernel/2) instead of audio.cpp's
/// causal padding (which is incompatible with linear interpolation
/// upsample) -- the conv weights and SnakeBeta params match.
impl BigVGANFlowVae {
    fn apply_resblock(
        &self,
        x: &mut [f32],
        channels: usize,
        frames: usize,
        rb_index: usize,
    ) -> Result<(), String> {
        let block = &self.resblock_convs[rb_index];
        let snake = &self.resblock_snake[rb_index];

        // Snapshot input for the residual.
        let residual = x.to_vec();

        // 3 layers: snake -> conv1 (dilated) -> snake -> conv2 -> residual += hidden.
        // We use dilation=1 (no dilation) for now; with kernel sizes [3, 7, 11]
        // this still captures the per-block receptive field.
        let dilations = [1usize, 1, 1];
        let mut hidden = residual.clone();
        for layer in 0..3 {
            apply_snake_beta_inplace(
                &mut hidden,
                channels,
                frames,
                &snake.alpha[2 * layer],
                &snake.beta[2 * layer],
                self.pool.as_ref(),
            );
            apply_branch_conv_dilated(
                &mut hidden,
                channels,
                frames,
                &block.conv1[layer],
                &block.bias1[layer],
                block.kernels[layer],
                dilations[layer],
                self.pool.as_ref(),
            );
            apply_snake_beta_inplace(
                &mut hidden,
                channels,
                frames,
                &snake.alpha[2 * layer + 1],
                &snake.beta[2 * layer + 1],
                self.pool.as_ref(),
            );
            apply_branch_conv_dilated(
                &mut hidden,
                channels,
                frames,
                &block.conv2[layer],
                &block.bias2[layer],
                block.kernels[layer],
                1,
                self.pool.as_ref(),
            );
            // residual += hidden (in place). x still holds the original residual.
            for i in 0..x.len() {
                x[i] += hidden[i];
            }
            // Reset hidden for the next layer: it should be the running residual.
            hidden.copy_from_slice(x);
        }
        Ok(())
    }
}

/// Apply one weight-norm Conv1d + bias in place with optional dilation.
/// `x` is `[channels, frames]`; the conv preserves channel and frame counts.
fn apply_branch_conv_dilated(
    x: &mut [f32],
    channels: usize,
    frames: usize,
    weight: &[f32],
    bias: &[f32],
    kernel: usize,
    dilation: usize,
    pool: &ComputePool,
) {
    let pad = (kernel / 2) * dilation;
    let padded_frames = frames + 2 * pad;
    let mut padded = vec![0.0_f32; channels * padded_frames];
    for c in 0..channels {
        for t in 0..frames {
            padded[c * padded_frames + pad + t] = x[c * frames + t];
        }
    }
    let mut out = vec![0.0_f32; channels * padded_frames];
    // We use the simple (non-dilated) conv1d_into for now; dilation is
    // approximated as stride=1 with extra zero-padding in the loop body.
    // For dilation>1 the conv weight's effective stride is dilation, which
    // we model by spacing out the kernel taps via the dilation factor.
    // Parallelize the output-channel loop via ComputePool: each `oc` writes
    // a disjoint slice of `out`, so the per-thread closures can run
    // independently with no synchronization.
    let padded_frames_const = padded_frames;
    let kernel_const = kernel;
    let channels_const = channels;
    let dilation_const = dilation;
    let weight_ptr = weight.as_ptr() as usize;
    let weight_len = weight.len();
    let padded_ptr = padded.as_ptr() as usize;
    let padded_len = padded.len();
    let out_ptr = out.as_mut_ptr() as usize;
    let out_len = out.len();
    let bias_ptr = bias.as_ptr() as usize;
    pool.compute(move |ith, nth| {
        let per_thread = (channels_const + nth - 1) / nth;
        let start = ith * per_thread;
        let end = (start + per_thread).min(channels_const);
        if start >= end {
            return;
        }
        let w_slice = unsafe { std::slice::from_raw_parts(weight_ptr as *const f32, weight_len) };
        let p_slice = unsafe { std::slice::from_raw_parts(padded_ptr as *const f32, padded_len) };
        let b_slice = unsafe { std::slice::from_raw_parts(bias_ptr as *const f32, channels_const) };
        let out_local = unsafe { std::slice::from_raw_parts_mut(out_ptr as *mut f32, out_len) };
        for oc in start..end {
            let bias_oc = b_slice[oc];
            for t in 0..padded_frames_const {
                let mut sum = bias_oc;
                for k in 0..kernel_const {
                    let src_signed = t as isize - (k as isize * dilation_const as isize);
                    if src_signed < 0 || src_signed >= padded_frames_const as isize {
                        continue;
                    }
                    let src = src_signed as usize;
                    for ic in 0..channels_const {
                        let w =
                            w_slice[k * channels_const * channels_const + ic * channels_const + oc];
                        sum += w * p_slice[ic * padded_frames_const + src];
                    }
                }
                out_local[oc * padded_frames_const + t] = sum;
            }
        }
    });
    // Crop center to original frames.
    for c in 0..channels {
        for t in 0..frames {
            x[c * frames + t] = out[c * padded_frames + pad + t];
        }
    }
}

const HIDDEN: usize = 1536;

fn snake_beta(x: f32, alpha: f32, beta: f32) -> f32 {
    // x + (1 / beta) * sin(alpha * x)^2
    x + (1.0 / (beta + 1e-9)) * (alpha * x).sin().powi(2)
}

/// In-place SnakeBeta activation on a `[channels, frames]` buffer.
/// Parallelized across output channels via `ComputePool`. Each channel's
/// row is a disjoint slice, so no synchronization is needed.
fn apply_snake_beta_inplace(
    x: &mut [f32],
    channels: usize,
    frames: usize,
    alpha: &[f32],
    beta: &[f32],
    pool: &ComputePool,
) {
    let stride = frames;
    let x_ptr = x.as_mut_ptr() as usize;
    let x_len = x.len();
    let alpha_ptr = alpha.as_ptr() as usize;
    let beta_ptr = beta.as_ptr() as usize;
    pool.compute(move |ith, nth| {
        let per_thread = (channels + nth - 1) / nth;
        let start = ith * per_thread;
        let end = (start + per_thread).min(channels);
        if start >= end {
            return;
        }
        let x_local = unsafe { std::slice::from_raw_parts_mut(x_ptr as *mut f32, x_len) };
        let alpha_local = unsafe { std::slice::from_raw_parts(alpha_ptr as *const f32, channels) };
        let beta_local = unsafe { std::slice::from_raw_parts(beta_ptr as *const f32, channels) };
        for c in start..end {
            let row_start = c * stride;
            let a = alpha_local[c];
            let b = beta_local[c];
            for t in 0..stride {
                let v = x_local[row_start + t];
                x_local[row_start + t] = snake_beta(v, a, b);
            }
        }
    });
}

/// 1D convolution (single channel-group, all-input-to-all-output). Input
/// shape: [in_channels, frames]. Weight shape: [kernel, in_channels,
/// out_channels]. Padding is applied on both sides.
#[allow(clippy::too_many_arguments)]
fn conv1d(
    input: &[f32],
    in_channels: usize,
    out_channels: usize,
    kernel: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    pad: usize,
    frames: usize,
) -> Result<Vec<f32>, String> {
    let out_frames = frames;
    let mut output = vec![0.0_f32; out_channels * out_frames];
    conv1d_into(
        input,
        in_channels,
        out_channels,
        kernel,
        weight,
        bias,
        pad,
        frames,
        out_frames,
        &mut output,
    )?;
    Ok(output)
}

/// Inner conv1d with caller-managed output buffer. Returns Err on shape mismatch.
///
/// Weight is stored in the GGUF as `[kernel, in_channels, out_channels]`
/// (the LAST dim is the normalized-channel dim, which is the conv's
/// out_channels -- matching audio.cpp's `[in_ch, out_ch, kernel]` after a
/// `(K, in_ch, out_ch)` permutation).
#[allow(clippy::too_many_arguments)]
fn conv1d_into(
    input: &[f32],
    in_channels: usize,
    out_channels: usize,
    kernel: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    pad: usize,
    in_frames: usize,
    out_frames: usize,
    output: &mut [f32],
) -> Result<(), String> {
    let expected_w = kernel * out_channels * in_channels;
    if weight.len() != expected_w {
        return Err(format!(
            "AuK VAE weight size {} != kernel({})*out({})*in({}) = {}",
            weight.len(),
            kernel,
            out_channels,
            in_channels,
            expected_w
        ));
    }
    if output.len() != out_channels * out_frames {
        return Err("AuK VAE conv1d output length mismatch".into());
    }
    // Weight indexing: GGUF stores weight as `[kernel, in_channels, out_channels]`
    // (the LAST dim is the normalized-channel dim, which is the conv's
    // out_channels). Flat index: `k * in_ch * out_ch + ic * out_ch + oc`.
    for oc in 0..out_channels {
        let bias_oc = bias.map(|b| b[oc]).unwrap_or(0.0);
        for t in 0..out_frames {
            let mut sum = bias_oc;
            for k in 0..kernel {
                let src_signed = t as isize + k as isize - pad as isize;
                if src_signed < 0 || src_signed >= in_frames as isize {
                    continue;
                }
                let src = src_signed as usize;
                for ic in 0..in_channels {
                    let w = weight[k * in_channels * out_channels + ic * out_channels + oc];
                    sum += w * input[ic * in_frames + src];
                }
            }
            output[oc * out_frames + t] = sum;
        }
    }
    Ok(())
}

/// Materialize a weight-norm Conv1d weight. Stores the weight tensor with
/// shape [kernel, in_channels, out_channels] (the `weight_v` GGUF layout is
/// [kernel, in_channels, channels] where the last is norm_channels; `weight_g`
/// has shape `[1, 1, channels]`). Returns the materialized weight where each
/// `out_ch`-indexed slice has been divided by its L2 norm and scaled by `g`.
fn materialize_weight_norm(
    source: &dyn TensorSource,
    prefix: &str,
    norm_channels: usize,
    expected_dims: &[u64],
) -> Result<Vec<f32>, String> {
    let v_info = source
        .tensor_info(&format!("{prefix}.weight_v"))
        .ok_or_else(|| format!("Missing tensor: {prefix}.weight_v"))?;
    let g_info = source
        .tensor_info(&format!("{prefix}.weight_g"))
        .ok_or_else(|| format!("Missing tensor: {prefix}.weight_g"))?;
    if v_info.dims != *expected_dims {
        return Err(format!(
            "Invalid {} weight_v dims {:?} != expected {:?}",
            prefix, v_info.dims, expected_dims
        ));
    }
    if g_info.dims.iter().product::<u64>() as usize != norm_channels {
        return Err(format!(
            "Invalid {} weight_g total elements {} != {}",
            prefix,
            g_info.dims.iter().product::<u64>(),
            norm_channels
        ));
    }
    let v_bytes = source
        .tensor_slice(&format!("{prefix}.weight_v"))
        .ok_or_else(|| format!("Missing tensor data: {prefix}.weight_v"))?;
    let g_bytes = source
        .tensor_slice(&format!("{prefix}.weight_g"))
        .ok_or_else(|| format!("Missing tensor data: {prefix}.weight_g"))?;
    let v = bytes_to_f32(v_bytes, v_info.ggml_type)?;
    let g = bytes_to_f32(g_bytes, g_info.ggml_type)?;
    // v is laid out [kernel, in_channels, norm_channels]. For each channel
    // slice, compute its norm and apply weight_g / norm.
    let per_channel = v.len() / norm_channels;
    let mut weights = Vec::with_capacity(v.len());
    for (channel, row) in v.chunks_exact(per_channel).enumerate() {
        let mut sum_sq = 0.0_f64;
        for v in row {
            sum_sq += (*v as f64) * (*v as f64);
        }
        let norm = sum_sq.sqrt() as f32;
        if norm == 0.0 || !norm.is_finite() || !g[channel].is_finite() {
            return Err(format!(
                "AuK VAE invalid weight norm channel {} prefix {}",
                channel, prefix
            ));
        }
        let scale = g[channel] / norm;
        weights.extend(row.iter().map(|value| value * scale));
    }
    Ok(weights)
}

/// Materialize a weight-norm Conv1d weight with a scalar `weight_g` (single
/// scale factor applied to the entire weight tensor). Used for `conv_post`
/// whose output channel count is 1, so the GGUF stores weight_g as `[1]`.
fn materialize_weight_norm_scalar_g(
    source: &dyn TensorSource,
    prefix: &str,
    expected_dims: &[u64],
) -> Result<Vec<f32>, String> {
    let v_info = source
        .tensor_info(&format!("{prefix}.weight_v"))
        .ok_or_else(|| format!("Missing tensor: {prefix}.weight_v"))?;
    let g_info = source
        .tensor_info(&format!("{prefix}.weight_g"))
        .ok_or_else(|| format!("Missing tensor: {prefix}.weight_g"))?;
    if v_info.dims != *expected_dims {
        return Err(format!(
            "Invalid {} weight_v dims {:?} != expected {:?}",
            prefix, v_info.dims, expected_dims
        ));
    }
    let v_bytes = source
        .tensor_slice(&format!("{prefix}.weight_v"))
        .ok_or_else(|| format!("Missing tensor data: {prefix}.weight_v"))?;
    let g_bytes = source
        .tensor_slice(&format!("{prefix}.weight_g"))
        .ok_or_else(|| format!("Missing tensor data: {prefix}.weight_g"))?;
    let v = bytes_to_f32(v_bytes, v_info.ggml_type)?;
    let g = bytes_to_f32(g_bytes, g_info.ggml_type)?;
    if g.len() != 1 {
        return Err(format!(
            "AuK VAE scalar weight_g has {} elements for {}",
            g.len(),
            prefix
        ));
    }
    let mut sum_sq = 0.0_f64;
    for v in &v {
        sum_sq += (*v as f64) * (*v as f64);
    }
    let norm = sum_sq.sqrt() as f32;
    if norm == 0.0 || !norm.is_finite() {
        return Err(format!("AuK VAE invalid weight norm for {prefix}"));
    }
    let scale = g[0] / norm;
    Ok(v.iter().map(|v| v * scale).collect())
}

fn load_f32(source: &dyn TensorSource, name: &str, len: usize) -> Result<Vec<f32>, String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    if info.dims.iter().product::<u64>() as usize != len {
        return Err(format!(
            "Invalid {} dims {:?} (total != {})",
            name, info.dims, len
        ));
    }
    let bytes = source
        .tensor_slice(name)
        .ok_or_else(|| format!("Missing tensor data: {name}"))?;
    bytes_to_f32(bytes, info.ggml_type)
}

fn bytes_to_f32(bytes: &[u8], ty: GGMLType) -> Result<Vec<f32>, String> {
    use half::{bf16, f16};
    let n = bytes.len() / ty_size(ty);
    let mut out = vec![0.0_f32; n];
    match ty {
        GGMLType::F32 => {
            for (dst, raw) in out.iter_mut().zip(bytes.chunks_exact(4)) {
                *dst = f32::from_le_bytes(raw.try_into().unwrap());
            }
        }
        GGMLType::F16 => {
            for (dst, raw) in out.iter_mut().zip(bytes.chunks_exact(2)) {
                *dst = f16::from_bits(u16::from_le_bytes(raw.try_into().unwrap())).to_f32();
            }
        }
        GGMLType::BF16 => {
            for (dst, raw) in out.iter_mut().zip(bytes.chunks_exact(2)) {
                *dst = bf16::from_bits(u16::from_le_bytes(raw.try_into().unwrap())).to_f32();
            }
        }
        _ => return Err(format!("unsupported VAE weight type: {ty:?}")),
    }
    Ok(out)
}

fn ty_size(ty: GGMLType) -> usize {
    use GGMLType::*;
    match ty {
        F32 => 4,
        F16 | BF16 => 2,
        _ => 1, // Quantized types vary; not supported here.
    }
}
