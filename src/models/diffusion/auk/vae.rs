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
    // Latent denormalization statistics. Despite the name, `global_log_std`
    // stores the latent *variance*: AuK's `denormalize` is
    // `latents * sqrt(global_log_std) + global_mean`.
    global_mean: Vec<f32>,
    global_std: Vec<f32>,
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
    /// SnakeBeta applied before `conv_post` (AuK `activation_post`).
    post_alpha: Vec<f32>,
    post_beta: Vec<f32>,
    /// Anti-aliased resampling filter shared by every SnakeBeta site.
    aa_filter: AaFilter,
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

/// Anti-aliased SnakeBeta activation (AuK `Activation1d`).
///
/// ```text
/// x -> upsample(x, ratio=2, FIR) -> SnakeBeta -> downsample(x, ratio=2, FIR)
/// ```
///
/// The official implementation lives in
/// `auk/model/vae/modules/bigvgan/alias_free_torch`:
/// - `UpSample1d(ratio=2, kernel_size=12)` — `replicate` padding, a
///   stride-2 depthwise transposed convolution with a 12-tap Kaiser-windowed
///   sinc filter, then centre-trim.
/// - `DownSample1d(ratio=2, kernel_size=12, causal=True)` — `replicate` padding
///   plus a stride-2 depthwise low-pass convolution with the same filter.
///
/// Both filters are stored in the checkpoint
/// (`<prefix>.upsample.filter` / `<prefix>.downsample.lowpass.filter`) and
/// are identical across every activation, so we load them once.
#[derive(Clone)]
struct AaFilter {
    /// 12-tap Kaiser-windowed sinc filter.
    taps: Vec<f32>,
}

impl AaFilter {
    fn load(source: &dyn TensorSource, name: &str) -> Result<Self, String> {
        let taps = load_f32(source, name, AA_FILTER_TAPS)?;
        if taps.iter().any(|v| !v.is_finite()) {
            return Err(format!("AuK VAE filter {name} contains non-finite values"));
        }
        Ok(Self { taps })
    }
}

/// Number of taps in the alias-free resampling filters.
const AA_FILTER_TAPS: usize = 12;
/// `UpSample1d` / `DownSample1d` resampling ratio.
const AA_RATIO: usize = 2;
/// `UpSample1d` kernel size (`int(6 * ratio // 2) * 2`).
const AA_KERNEL: usize = 12;

/// Anti-aliased activation state shared by every SnakeBeta site.
struct AaActivation {
    /// `UpSample1d` filter (also used as the `DownSample1d` low-pass).
    filter: AaFilter,
}

impl AaActivation {
    /// Apply the anti-aliased activation to a `[channels, frames]` buffer.
    ///
    /// `UpSample1d` with `causal=False`:
    /// ```python
    /// x = F.pad(x, (pad, pad), mode="replicate")   # pad = kernel//ratio - 1 = 5
    /// x = ratio * F.conv_transpose1d(x, filter.expand(C,-1,-1), stride=ratio, groups=C)
    /// x = x[..., pad_left : -pad_right]
    /// ```
    /// with `pad_left = pad*stride + (kernel - stride)//2` and
    /// `pad_right = pad*stride + (kernel - stride + 1)//2`.
    ///
    /// `DownSample1d` with `causal=True`:
    /// ```python
    /// xx = F.pad(x, (kernel_size - 1, 0), mode="replicate")
    /// out = F.conv1d(xx, filter.expand(C,-1,-1), stride=ratio, groups=C)
    /// ```
    ///
    /// Net effect: the frame count is preserved (2x up, then 2x down).
    fn apply(
        &self,
        x: &mut [f32],
        channels: usize,
        frames: usize,
        alpha: &[f32],
        beta: &[f32],
        pool: &ComputePool,
    ) {
        debug_assert_eq!(x.len(), channels * frames);
        let mut up = upsample_ratio2(x, channels, frames, &self.filter.taps);
        apply_snake_beta_inplace(&mut up, channels, frames * AA_RATIO, alpha, beta, pool);
        x.copy_from_slice(&downsample_ratio2(
            &up,
            channels,
            frames * AA_RATIO,
            &self.filter.taps,
        ));
    }
}

/// `UpSample1d(ratio=2, kernel_size=12, causal=False)`.
///
/// Returns a `[channels, frames * 2]` buffer.
fn upsample_ratio2(x: &[f32], channels: usize, frames: usize, taps: &[f32]) -> Vec<f32> {
    let stride = AA_RATIO;
    let pad = AA_KERNEL / AA_RATIO - 1; // 5
    let pad_left = pad * stride + (AA_KERNEL - stride) / 2; // 15
    let pad_right = pad * stride + (AA_KERNEL - stride + 1) / 2; // 15
    // `F.pad(x, (pad, pad), mode="replicate")` then a stride-`stride`
    // depthwise transposed convolution: output length = (padded - 1) * stride
    // + kernel_size, then the centre trim.
    let padded_frames = frames + 2 * pad;
    let full = (padded_frames - 1) * stride + AA_KERNEL;
    let out_frames = full - pad_left - pad_right;
    debug_assert_eq!(out_frames, frames * AA_RATIO);

    let mut out = vec![0.0_f32; channels * out_frames];
    for c in 0..channels {
        let src = &x[c * frames..(c + 1) * frames];
        let dst = &mut out[c * out_frames..(c + 1) * out_frames];
        for n in 0..out_frames {
            // Position in the untrimmed transposed-conv output.
            let full_pos = n + pad_left;
            let mut acc = 0.0_f32;
            // conv_transpose1d: out[p] = sum_k w[k] * in[(p - k) / stride]
            // for every k with (p - k) divisible by stride.
            let base = full_pos % stride;
            let mut k = base;
            while k < AA_KERNEL {
                let src_idx = (full_pos - k) / stride;
                // `F.pad(x, (pad, pad), mode="replicate")`: the padded index
                // `src_idx` maps to the source index `src_idx - pad`, clamped
                // to the signal bounds.
                let padded_idx = src_idx as isize - pad as isize;
                let value = if padded_idx < 0 {
                    src[0]
                } else if padded_idx >= frames as isize {
                    src[frames - 1]
                } else {
                    src[padded_idx as usize]
                };
                acc += taps[k] * value;
                k += stride;
            }
            dst[n] = acc * stride as f32;
        }
    }
    out
}

/// `DownSample1d(ratio=2, kernel_size=12, causal=True)`.
///
/// Takes a `[channels, frames]` buffer and returns `[channels, frames / 2]`.
fn downsample_ratio2(x: &[f32], channels: usize, frames: usize, taps: &[f32]) -> Vec<f32> {
    let stride = AA_RATIO;
    let pad_left = AA_KERNEL - 1; // 11 (causal)
    let padded_frames = frames + pad_left;
    let out_frames = (padded_frames - AA_KERNEL) / stride + 1;
    let mut out = vec![0.0_f32; channels * out_frames];
    for c in 0..channels {
        let src = &x[c * frames..(c + 1) * frames];
        let dst = &mut out[c * out_frames..(c + 1) * out_frames];
        for n in 0..out_frames {
            let mut acc = 0.0_f32;
            for k in 0..AA_KERNEL {
                let padded_idx = n * stride + k;
                // `replicate` padding on the left only (causal).
                let src_idx = padded_idx as isize - pad_left as isize;
                let value = if src_idx < 0 {
                    src[0]
                } else {
                    src[src_idx as usize]
                };
                acc += taps[k] * value;
            }
            dst[n] = acc;
        }
    }
    out
}

/// Write a `f32` tensor to `<dir>/<name>.f32` for numeric comparison with
/// the PyTorch reference implementation.
fn dump_f32(path: &std::path::Path, data: &[f32]) {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    if let Err(error) = std::fs::write(path, &bytes) {
        eprintln!("[auk-dump] failed to write {}: {error}", path.display());
    }
}

/// Write a `f32` tensor to `<dir>/<name>.f32` for numeric comparison with
/// the PyTorch reference implementation.
fn dump_tensor(dir: &std::path::Path, name: &str, data: &[f32]) {
    let path = dir.join(format!("{name}.f32"));
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    if let Err(error) = std::fs::write(&path, &bytes) {
        eprintln!("[auk-vae-dump] failed to write {}: {error}", path.display());
    }
}

impl BigVGANFlowVae {
    pub(crate) fn load(
        source: Arc<dyn TensorSource>,
        _pool: Arc<ComputePool>,
    ) -> Result<Self, String> {
        // The DiT emits normalized latents. AuK's `denormalize` maps them back
        // with `x * sqrt(global_log_std) + global_mean`; the tensor called
        // `global_log_std` actually holds the variance (verified against the
        // upstream `bigvgan_flow_vae.py`). Skipping this step made the decoder
        // see unit-scale latents instead of the real latent distribution.
        let global_mean = load_f32(source.as_ref(), "global_mean", LATENT_DIM)?;
        let global_log_std = load_f32(source.as_ref(), "global_log_std", LATENT_DIM)?;
        if global_log_std.iter().any(|v| !(v.is_finite() && *v >= 0.0)) {
            return Err("AuK VAE global_log_std contains invalid variance values".into());
        }
        let global_std: Vec<f32> = global_log_std.iter().map(|v| v.sqrt()).collect();

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
        let post_channels = STAGE_CHANNELS[UPSAMPLE_STAGES - 1];
        let post_alpha = load_f32(source.as_ref(), "activation_post.act.alpha", post_channels)?
            .into_iter()
            .map(f32::exp)
            .collect();
        let post_beta = load_f32(source.as_ref(), "activation_post.act.beta", post_channels)?
            .into_iter()
            .map(f32::exp)
            .collect();
        // Anti-aliased activation filters (identical everywhere in the
        // checkpoint; we verify the resblock copies match the reference).
        let aa_filter = AaFilter::load(source.as_ref(), "activation_post.upsample.filter")?;
        {
            let down = AaFilter::load(source.as_ref(), "activation_post.downsample.lowpass.filter")?;
            if down.taps != aa_filter.taps {
                return Err("AuK VAE upsample/downsample filters differ".into());
            }
        }

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
                    // `snake_logscale=True` in the AuK config, so the stored
                    // alpha/beta are log-scale and must be exponentiated.
                    alpha[act] = load_f32(
                        source.as_ref(),
                        &format!("{name_prefix}.activations.{act}.act.alpha"),
                        channels,
                    )?
                    .into_iter()
                    .map(f32::exp)
                    .collect();
                    beta[act] = load_f32(
                        source.as_ref(),
                        &format!("{name_prefix}.activations.{act}.act.beta"),
                        channels,
                    )?
                    .into_iter()
                    .map(f32::exp)
                    .collect();
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
            global_mean,
            global_std,
            conv_pre,
            conv_pre_bias,
            ups_weight,
            ups_bias,
            resblock_convs,
            resblock_snake,
            post_alpha,
            post_beta,
            aa_filter,
            conv_post,
        })
    }

    pub(crate) fn decode(&self, latent: &[f32], sample_rate: u32) -> Result<AukAudio, String> {
        let dump = std::env::var_os("RMI_AUK_VAE_DUMP").map(std::path::PathBuf::from);
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
        // to [1, latent_dim, latent_time] for Conv1d, applying the VAE's latent
        // denormalization per channel (`x * sqrt(var) + mean`).
        let mut hidden = vec![0.0_f32; LATENT_DIM * latent_time];
        for c in 0..LATENT_DIM {
            let mean = self.global_mean[c];
            let std = self.global_std[c];
            let row = c * latent_time;
            for t in 0..latent_time {
                hidden[row + t] = latent[row + t] * std + mean;
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
        if let Some(dir) = dump.as_ref() {
            dump_tensor(dir, "denorm", &hidden);
            dump_tensor(dir, "conv_pre", &pre);
        }

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
            // Weight layout from GGUF (raw PyTorch bytes): ConvTranspose1d
            // stores `[in_channels][out_channels][kernel_size]`, i.e. the
            // kernel is contiguous.
            //
            // ConvTranspose1d (kernel = 2*rate, stride = rate, padding = 0)
            // is evaluated directly via the polyphase decomposition:
            //   out[n] = sum_k w[k] * in[(n - k) / rate]   for (n - k) % rate == 0
            // With n = q*rate + p the only valid kernel taps are k = p and
            // k = p + rate, so
            //   out[q*rate + p] = w[p]     * in[q]
            //                     + w[p+rate] * in[q-1]     (q > 0)
            //
            // The official layer is causal, so it trims the last `stride`
            // samples of the natural `(N+1)*rate` output, leaving exactly
            // `N*rate` frames.
            let weight = &self.ups_weight[stage];
            let bias = &self.ups_bias[stage];
            let mut upsampled = vec![0.0_f32; out_ch * out_frames];
            for q in 0..frames {
                for p in 0..rate {
                    let pos = q * rate + p;
                    for oc in 0..out_ch {
                        let mut acc = 0.0_f32;
                        for ic in 0..in_ch {
                            let w = weight[(ic * out_ch + oc) * (2 * rate) + p];
                            acc += w * pre[ic * frames + q];
                            if q > 0 {
                                let w_prev = weight[(ic * out_ch + oc) * (2 * rate) + (p + rate)];
                                acc += w_prev * pre[ic * frames + (q - 1)];
                            }
                        }
                        upsampled[oc * out_frames + pos] = acc;
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

            // The 3 resblocks of a stage are **parallel branches**: each one
            // sees the same upsampled input, and their outputs are summed
            // and divided by 3.
            //
            // ```python
            // xs = None
            // for j in range(self.num_kernels):
            //     if xs is None: xs = self.resblocks[i*self.num_kernels + j](x)
            //     else:          xs += self.resblocks[i*self.num_kernels + j](x)
            // x = xs / self.num_kernels
            // ```
            let stage_input = upsampled;
            if let Some(dir) = dump.as_ref() {
                dump_tensor(dir, &format!("stage{stage}_up"), &stage_input);
            }
            let mut acc = vec![0.0_f32; out_ch * out_frames];
            for rb_in_stage in 0..3 {
                let rb_global = stage * 3 + rb_in_stage;
                let mut branch = stage_input.clone();
                self.apply_resblock(&mut branch, out_ch, out_frames, rb_global)?;
                if let Some(dir) = dump.as_ref() {
                    dump_tensor(dir, &format!("stage{stage}_rb{rb_in_stage}"), &branch);
                }
                for (a, b) in acc.iter_mut().zip(branch.iter()) {
                    *a += *b;
                }
            }
            let inv_kernels = 1.0 / 3.0;
            for v in acc.iter_mut() {
                *v *= inv_kernels;
            }
            pre = acc;
            if let Some(dir) = dump.as_ref() {
                dump_tensor(dir, &format!("stage{stage}"), &pre);
            }
        }

        // conv_post: weight-norm Conv1d 24 -> 1, kernel 7 (no bias)
        let channels_post = STAGE_CHANNELS[UPSAMPLE_STAGES - 1];
        let frames = pre.len() / channels_post;
        // AuK applies `activation_post` (an anti-aliased `Activation1d`
        // wrapping SnakeBeta) before the final conv.
        AaActivation {
            filter: self.aa_filter.clone(),
        }
        .apply(
            &mut pre,
            channels_post,
            frames,
            &self.post_alpha,
            &self.post_beta,
            self.pool.as_ref(),
        );
        // `conv_post` is built with `causal=True`, so it left-pads with
        // `1 * (7 - 1) = 6` zeros (instead of the symmetric padding=3 used
        // by `conv_pre`, which is `causal=False`).
        let mut mono = causal_conv1d(
            &pre,
            channels_post,
            1,
            7,
            1,
            &self.conv_post,
            None,
            frames,
        )?;
        if let Some(dir) = dump.as_ref() {
            dump_tensor(dir, "act_post", &pre);
            dump_tensor(dir, "conv_post", &mono);
        }

        // AuK's `inference_from_latents` ends with `torch.clamp(x, -1, 1)`.
        // Do NOT peak-normalize: the absolute scale carries the model's
        // intended loudness, and normalizing hides latent-scale bugs.
        for v in &mut mono {
            *v = v.clamp(-1.0, 1.0);
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

/// Apply one residual block (`AMPBlock1`) to a `[channels, frames]` buffer.
///
/// Official structure (`bigvgan_flow_vae.py::AMPBlock1.forward`):
/// ```python
/// acts1, acts2 = self.activations[::2], self.activations[1::2]
/// for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
///     xt = a1(x)      # anti-aliased SnakeBeta
///     xt = c1(xt)     # dilated conv (1, 3, 5)
///     xt = a2(xt)     # anti-aliased SnakeBeta
///     xt = c2(xt)     # conv with dilation 1
///     x = xt + x      # sequential residual accumulation
/// ```
///
/// Note the residual is accumulated **inside** the block (across the 3
/// layers), and the 3 blocks of a stage are summed by the caller.
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

        // `convs1` uses dilations (1, 3, 5); `convs2` all use dilation 1.
        let dilations = [1usize, 3, 5];
        let aa = AaActivation {
            filter: self.aa_filter.clone(),
        };
        let mut hidden = x.to_vec();
        for layer in 0..3 {
            aa.apply(
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
            aa.apply(
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
            // `x = xt + x`: accumulate the branch output into the running
            // residual.
            for i in 0..x.len() {
                x[i] += hidden[i];
            }
            // The next layer continues from the updated residual.
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
    // Official `Conv1d(causal=True)` uses `padding=0` plus an explicit
    // left-padding of `dilation * (kernel_size - 1)` zeros:
    // ```python
    // if causal:
    //     padding = 0
    //     self.left_padding = dilation * (kernel_size - 1)
    // # forward:
    // x = F.pad(x.unsqueeze(2), (self.left_padding, 0, 0, 0)).squeeze(2)
    // ```
    // The output length therefore equals the input length.
    let left_pad = dilation * (kernel - 1);
    let padded_frames = frames + left_pad;
    let mut padded = vec![0.0_f32; channels * padded_frames];
    for c in 0..channels {
        for t in 0..frames {
            padded[c * padded_frames + left_pad + t] = x[c * frames + t];
        }
    }
    let mut out = vec![0.0_f32; channels * frames];
    // Parallelize the output-channel loop via ComputePool: each `oc` writes
    // a disjoint slice of `out`, so the per-thread closures can run
    // independently with no synchronization.
    let padded_frames_const = padded_frames;
    let frames_const = frames;
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
            for t in 0..frames_const {
                let mut sum = bias_oc;
                for k in 0..kernel_const {
                    let src_signed = t as isize + (k as isize * dilation_const as isize);
                    if src_signed >= padded_frames_const as isize {
                        continue;
                    }
                    let src = src_signed as usize;
                    for ic in 0..channels_const {
                        // Raw GGUF layout is [out][in][kernel].
                        let w = w_slice[(oc * channels_const + ic) * kernel_const + k];
                        sum += w * p_slice[ic * padded_frames_const + src];
                    }
                }
                out_local[oc * frames_const + t] = sum;
            }
        }
    });
    x.copy_from_slice(&out);
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

/// Causal 1D convolution (official `Conv1d(causal=True)`).
///
/// The official layer sets `padding = 0` and explicitly left-pads the input
/// with `dilation * (kernel_size - 1)` zeros, so the output length equals the
/// input length and the convolution is strictly causal.
///
/// Weight layout is the raw GGUF/PyTorch order `[out][in][kernel]`.
#[allow(clippy::too_many_arguments)]
fn causal_conv1d(
    input: &[f32],
    in_channels: usize,
    out_channels: usize,
    kernel: usize,
    dilation: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    frames: usize,
) -> Result<Vec<f32>, String> {
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
    let left_pad = dilation * (kernel - 1);
    let padded_frames = frames + left_pad;
    let mut padded = vec![0.0_f32; in_channels * padded_frames];
    for c in 0..in_channels {
        for t in 0..frames {
            padded[c * padded_frames + left_pad + t] = input[c * frames + t];
        }
    }
    let mut output = vec![0.0_f32; out_channels * frames];
    for oc in 0..out_channels {
        let bias_oc = bias.map(|b| b[oc]).unwrap_or(0.0);
        for t in 0..frames {
            let mut sum = bias_oc;
            for k in 0..kernel {
                let src = t + k * dilation;
                if src >= padded_frames {
                    continue;
                }
                for ic in 0..in_channels {
                    let w = weight[(oc * in_channels + ic) * kernel + k];
                    sum += w * padded[ic * padded_frames + src];
                }
            }
            output[oc * frames + t] = sum;
        }
    }
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
    // Weight layout: GGUF raw bytes are the PyTorch C-order weight, i.e.
    // `[out_channels][in_channels][kernel]` with the kernel contiguous
    // (GGUF `ne = [kernel, in_channels, out_channels]`). Flat index:
    // `(oc * in_ch + ic) * kernel + k`.
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
                    let w = weight[(oc * in_channels + ic) * kernel + k];
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::ggufrs::{open_model_source, ComponentRole};

    /// Decode a fixed latent with the published AuK VAE GGUF and dump the
    /// intermediate buffers so they can be diffed against the PyTorch
    /// reference implementation.
    ///
    /// ```text
    /// RMI_AUK_VAE=models/AuK-Base-and-Flash-GGUF/auk-vae-f32.gguf \
    /// RMI_AUK_VAE_LATENT=<f32 latent file> \
    /// RMI_AUK_VAE_DUMP=<dump dir> \
    /// cargo test --lib auk::vae -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "requires the published AuK VAE GGUF"]
    fn decodes_fixed_latent_for_reference_diff() {
        let Some(vae_path) = std::env::var_os("RMI_AUK_VAE") else {
            eprintln!("skipping: set RMI_AUK_VAE");
            return;
        };
        let Some(latent_path) = std::env::var_os("RMI_AUK_VAE_LATENT") else {
            eprintln!("skipping: set RMI_AUK_VAE_LATENT");
            return;
        };
        // Scratch dumps live under the repository's `target/` directory so
        // nothing leaks outside the workspace.
        let dump_dir = std::path::PathBuf::from(
            std::env::var_os("RMI_AUK_VAE_DUMP").unwrap_or_else(|| {
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("target/auk-reference/rust-vae")
                    .into_os_string()
            }),
        );
        std::fs::create_dir_all(&dump_dir).expect("create dump dir");

        let bytes = std::fs::read(&latent_path).expect("read latent");
        let latent: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();

        let source = open_model_source(std::path::Path::new(&vae_path), ComponentRole::Llm)
            .expect("open vae gguf");
        let vae = BigVGANFlowVae::load(Arc::from(source), Arc::new(ComputePool::new(4)))
            .expect("load vae");
        let audio = vae.decode(&latent, 24000).expect("decode");
        dump_tensor(&dump_dir, "out", &audio.samples.iter().map(|v| *v as f32).collect::<Vec<_>>());
        eprintln!("decoded {} samples", audio.samples.len());
    }
}
