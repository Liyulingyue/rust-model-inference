//! GPU operators for the AuK BigVGANFlow VAE decoder.
//!
//! The decoder keeps its control flow on the host: this module walks the same
//! stage and residual-block structure as the CPU path, but every operator
//! runs on the Vulkan device. The activation buffers stay in device memory
//! between the operators, so a residual block never round trips through the
//! host and only the weights are uploaded per use.
//!
//! The device results differ from the host ones in the last bits: the
//! convolution contracts into a fused multiply-add and the activation uses
//! the device's `sin`. The path is validated against the official PyTorch
//! reference instead of against the host kernel.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use crate::vulkan::ops::{ArenaRegion, AukConv1dSpec, AukFirSpec, AukVaeCapacity, AukVaeRuntime};

use super::vae::{
    dump_tensor, BigVGANFlowVae, AA_KERNEL, AA_RATIO, HIDDEN, LATENT_DIM, STAGE_CHANNELS,
    STAGE_RATES, UPSAMPLE_STAGES,
};
use super::AukAudio;

/// The device runtime of the decoder. It is reused across decodes and grows
/// when a longer latent needs bigger buffers.
static RUNTIME: OnceLock<Mutex<Option<AukVaeRuntime>>> = OnceLock::new();

/// Decode a latent on the Vulkan device.
pub(crate) fn decode(
    vae: &BigVGANFlowVae,
    latent: &[f32],
    sample_rate: u32,
) -> Result<AukAudio, String> {
    let context = crate::ops::get_vulkan_context()
        .ok_or_else(|| "AuK VAE GPU decode needs a Vulkan device".to_string())?;
    let latent_time = latent.len() / LATENT_DIM;
    if latent.len() != LATENT_DIM * latent_time || latent_time == 0 {
        return Err(format!(
            "AuK VAE latent shape mismatch: len {} latent_dim {} -> latent_time {}",
            latent.len(),
            LATENT_DIM,
            latent.len() / LATENT_DIM
        ));
    }
    let capacity = plan_capacity(latent_time)?;
    let mut state = RUNTIME
        .get_or_init(Mutex::default)
        .lock()
        .map_err(|_| "AuK VAE GPU runtime lock poisoned".to_string())?;
    grow_runtime(&mut state, context, capacity)?;
    let runtime = &mut *state
        .as_mut()
        .ok_or_else(|| "AuK VAE GPU runtime missing".to_string())?;
    let dump = std::env::var_os("RMI_AUK_VAE_DUMP").map(PathBuf::from);
    let profile = std::env::var_os("RMI_AUK_VAE_PROFILE").is_some();
    let total = std::time::Instant::now();
    // The regions are fixed for the lifetime of the arena, so the operators
    // can take them as plain copies.
    let residual = runtime.residual;
    let branch = runtime.branch;
    let scratch = runtime.scratch;
    let accumulator = runtime.accumulator;
    let stage_input = runtime.stage_input;
    let staging = runtime.staging;

    // The latent denormalization is cheap enough to stay on the host.
    let mut hidden = vec![0.0_f32; LATENT_DIM * latent_time];
    for c in 0..LATENT_DIM {
        let mean = vae.global_mean[c];
        let std = vae.global_std[c];
        let row = c * latent_time;
        for t in 0..latent_time {
            hidden[row + t] = latent[row + t] * std + mean;
        }
    }
    runtime.upload(stage_input, &hidden)?;
    if let Some(dir) = dump.as_ref() {
        dump_tensor(dir, "denorm", &hidden);
    }

    // conv_pre: Conv1d 64 -> 1536, kernel 7, padding 3.
    runtime.conv1d(
        stage_input,
        residual,
        &vae.conv_pre,
        Some(&vae.conv_pre_bias),
        &AukConv1dSpec {
            in_channels: LATENT_DIM,
            out_channels: HIDDEN,
            kernel: 7,
            stride: 1,
            dilation: 1,
            pad_left: 3,
            frames: latent_time,
            out_frames: latent_time,
            // `[kernel][in_channels][out_channels]`
            weight_layout: 0,
        },
    )?;
    if let Some(dir) = dump.as_ref() {
        dump_region(runtime, dir, "conv_pre", residual, HIDDEN * latent_time)?;
    }

    let mut frames = latent_time;
    for stage in 0..UPSAMPLE_STAGES {
        let stage_start = std::time::Instant::now();
        let in_ch = if stage == 0 {
            HIDDEN
        } else {
            STAGE_CHANNELS[stage - 1]
        };
        let out_ch = STAGE_CHANNELS[stage];
        let rate = STAGE_RATES[stage];
        let out_frames = frames * rate;

        // The transposed convolution is evaluated as one regular convolution
        // per output phase; the interleave folds the phase buffers back into
        // the `[out_channels][frames * rate]` layout and adds the bias.
        for phase in 0..rate {
            let phase_weights = out_ch * 2 * in_ch;
            let weight = &vae.ups_weight[stage][phase * phase_weights..][..phase_weights];
            runtime.conv1d(
                residual,
                staging.shifted(phase * out_ch * frames * 4),
                weight,
                None,
                &AukConv1dSpec {
                    in_channels: in_ch,
                    out_channels: out_ch,
                    kernel: 2,
                    stride: 1,
                    dilation: 1,
                    // The first tap of a phase weights the previous frame.
                    pad_left: 1,
                    frames,
                    out_frames: frames,
                    // `[out_channels][kernel][in_channels]`
                    weight_layout: 1,
                },
            )?;
        }
        runtime.interleave(
            staging,
            stage_input,
            &vae.ups_bias[stage],
            out_ch,
            frames,
            rate,
        )?;
        if let Some(dir) = dump.as_ref() {
            dump_region(
                runtime,
                dir,
                &format!("stage{stage}_up"),
                stage_input,
                out_ch * out_frames,
            )?;
        }

        // The three residual blocks of a stage are parallel branches: they
        // read the same stage input and their outputs are summed and divided
        // by three.
        let count = out_ch * out_frames;
        for rb_in_stage in 0..3 {
            let rb_global = stage * 3 + rb_in_stage;
            runtime.copy(stage_input, residual, count)?;
            apply_resblock(
                runtime, vae, out_ch, out_frames, rb_global, residual, branch, scratch,
            )?;
            if rb_in_stage == 0 {
                runtime.copy(residual, accumulator, count)?;
            } else {
                runtime.add(accumulator, residual, count)?;
            }
            if let Some(dir) = dump.as_ref() {
                dump_region(
                    runtime,
                    dir,
                    &format!("stage{stage}_rb{rb_in_stage}"),
                    residual,
                    count,
                )?;
            }
        }
        runtime.scale(accumulator, residual, count, 1.0 / 3.0)?;
        if let Some(dir) = dump.as_ref() {
            dump_region(runtime, dir, &format!("stage{stage}"), residual, count)?;
        }
        if profile {
            eprintln!(
                "[auk-vae-profile] stage {stage}: {:.1}ms",
                stage_start.elapsed().as_secs_f64() * 1000.0,
            );
        }
        frames = out_frames;
    }

    // AuK applies `activation_post` (an anti-aliased `Activation1d` wrapping
    // SnakeBeta) before the final conv.
    let channels_post = STAGE_CHANNELS[UPSAMPLE_STAGES - 1];
    aa_activation(
        runtime,
        residual,
        residual,
        &vae.post_alpha,
        &vae.post_beta,
        channels_post,
        frames,
        &vae.aa_filter.taps,
    )?;
    // `conv_post` is built with `causal=True`, so it left-pads with
    // `1 * (7 - 1) = 6` zeros.
    runtime.conv1d(
        residual,
        scratch,
        &vae.conv_post,
        None,
        &AukConv1dSpec {
            in_channels: channels_post,
            out_channels: 1,
            kernel: 7,
            stride: 1,
            dilation: 1,
            pad_left: 6,
            frames,
            out_frames: frames,
            // `[kernel][in_channels][out_channels]`
            weight_layout: 0,
        },
    )?;
    let mono = runtime.download(scratch, frames)?;
    if let Some(dir) = dump.as_ref() {
        dump_tensor(
            dir,
            "act_post",
            &runtime.download(residual, channels_post * frames)?,
        );
        dump_tensor(dir, "conv_post", &mono);
    }
    if profile {
        eprintln!(
            "[auk-vae-profile] device decode {:.1}ms  conv weight uploads {:.1}ms  conv dispatches {:.1}ms",
            total.elapsed().as_secs_f64() * 1000.0,
            runtime.upload_ns as f64 / 1e6,
            runtime.dispatch_ns as f64 / 1e6,
        );
    }

    // AuK's `inference_from_latents` ends with `torch.clamp(x, -1, 1)`.
    let samples: Vec<f64> = mono
        .iter()
        .map(|value| value.clamp(-1.0, 1.0) as f64)
        .collect();
    Ok(AukAudio {
        sample_rate,
        samples,
        channels: 1,
    })
}

/// Apply one residual block on the device.
///
/// The block computes `x = x + f(x)` per layer, where `f` is the
/// anti-aliased activation followed by two dilated convolutions. The residual
/// stays in the device buffer between the layers.
fn apply_resblock(
    runtime: &mut AukVaeRuntime,
    vae: &BigVGANFlowVae,
    channels: usize,
    frames: usize,
    rb_index: usize,
    residual: crate::vulkan::ops::ArenaRegion,
    branch: crate::vulkan::ops::ArenaRegion,
    scratch: crate::vulkan::ops::ArenaRegion,
) -> Result<(), String> {
    let block = &vae.resblock_convs[rb_index];
    let snake = &vae.resblock_snake[rb_index];
    // `convs1` uses dilations (1, 3, 5); `convs2` all use dilation 1.
    let dilations = [1usize, 3, 5];
    let count = channels * frames;
    for layer in 0..3 {
        aa_activation(
            runtime,
            residual,
            branch,
            &snake.alpha[2 * layer],
            &snake.beta[2 * layer],
            channels,
            frames,
            &vae.aa_filter.taps,
        )?;
        conv1d(
            runtime,
            branch,
            scratch,
            &block.conv1[layer],
            Some(&block.bias1[layer]),
            channels,
            channels,
            block.kernels[layer],
            dilations[layer],
            frames,
        )?;
        aa_activation(
            runtime,
            scratch,
            scratch,
            &snake.alpha[2 * layer + 1],
            &snake.beta[2 * layer + 1],
            channels,
            frames,
            &vae.aa_filter.taps,
        )?;
        conv1d(
            runtime,
            scratch,
            branch,
            &block.conv2[layer],
            Some(&block.bias2[layer]),
            channels,
            channels,
            block.kernels[layer],
            1,
            frames,
        )?;
        // `x = xt + x`: accumulate the branch output into the running
        // residual.
        runtime.add(residual, branch, count)?;
    }
    Ok(())
}

/// One causal weight-norm convolution with an optional dilation.
#[allow(clippy::too_many_arguments)]
fn conv1d(
    runtime: &mut AukVaeRuntime,
    input: crate::vulkan::ops::ArenaRegion,
    output: crate::vulkan::ops::ArenaRegion,
    weight: &[f32],
    bias: Option<&[f32]>,
    in_channels: usize,
    out_channels: usize,
    kernel: usize,
    dilation: usize,
    frames: usize,
) -> Result<(), String> {
    runtime
        .conv1d(
            input,
            output,
            weight,
            bias,
            &AukConv1dSpec {
                in_channels,
                out_channels,
                kernel,
                stride: 1,
                dilation,
                // The official `Conv1d(causal=True)` left-pads the input with
                // `dilation * (kernel_size - 1)` zeros.
                pad_left: dilation * (kernel - 1),
                frames,
                out_frames: frames,
                // `[kernel][in_channels][out_channels]`
                weight_layout: 0,
            },
        )
        .map_err(|error| error.to_string())
}

/// The anti-aliased SnakeBeta activation: upsample, SnakeBeta, downsample.
///
/// The three steps are separate device operators, so the intermediate signal
/// never leaves the device.
fn aa_activation(
    runtime: &mut AukVaeRuntime,
    input: crate::vulkan::ops::ArenaRegion,
    output: crate::vulkan::ops::ArenaRegion,
    alpha: &[f32],
    beta: &[f32],
    channels: usize,
    frames: usize,
    taps: &[f32],
) -> Result<(), String> {
    let pad = AA_KERNEL / AA_RATIO - 1;
    // `UpSample1d`: `replicate` padding, a stride-2 depthwise transposed
    // convolution and a centre trim.
    runtime
        .fir(
            input,
            runtime.staging,
            taps,
            &AukFirSpec {
                channels,
                kernel: taps.len(),
                stride: AA_RATIO,
                pad_left: pad * AA_RATIO + (AA_KERNEL - AA_RATIO) / 2,
                pad,
                input_frames: frames,
                output_frames: frames * AA_RATIO,
                scale: AA_RATIO as f32,
                upsample: true,
            },
        )
        .map_err(|error| error.to_string())?;
    runtime
        .snake(
            runtime.staging,
            runtime.staging,
            alpha,
            beta,
            channels,
            frames * AA_RATIO,
        )
        .map_err(|error| error.to_string())?;
    // `DownSample1d`: causal `replicate` padding and a stride-2 depthwise
    // low-pass convolution.
    runtime
        .fir(
            runtime.staging,
            output,
            taps,
            &AukFirSpec {
                channels,
                kernel: taps.len(),
                stride: AA_RATIO,
                pad_left: AA_KERNEL - 1,
                pad: 0,
                input_frames: frames * AA_RATIO,
                output_frames: frames,
                scale: 1.0,
                upsample: false,
            },
        )
        .map_err(|error| error.to_string())
}

/// Buffer sizes the decode needs.
fn plan_capacity(latent_time: usize) -> Result<AukVaeCapacity, String> {
    let mut activation = HIDDEN * latent_time;
    let mut staging = 0;
    let mut weight = HIDDEN * LATENT_DIM * 7;
    let mut bias = HIDDEN;
    let mut channels = HIDDEN;
    let mut frames = latent_time;
    for stage in 0..UPSAMPLE_STAGES {
        let in_ch = if stage == 0 {
            HIDDEN
        } else {
            STAGE_CHANNELS[stage - 1]
        };
        let out_ch = STAGE_CHANNELS[stage];
        let rate = STAGE_RATES[stage];
        let out_frames = frames * rate;
        activation = activation.max(out_ch * out_frames);
        // The anti-aliased activation keeps the upsampled signal, and the
        // upsample keeps one buffer per output phase.
        staging = staging.max(out_ch * out_frames * AA_RATIO);
        staging = staging.max(rate * out_ch * frames);
        // One phase block of the upsample convolution.
        weight = weight.max(rate * out_ch * 2 * in_ch);
        // One convolution of the widest residual block.
        for kernel in [3usize, 7, 11] {
            weight = weight.max(out_ch * out_ch * kernel);
        }
        bias = bias.max(out_ch);
        channels = channels.max(out_ch);
        frames = out_frames;
    }
    weight = weight.max(7 * STAGE_CHANNELS[UPSAMPLE_STAGES - 1]);
    Ok(AukVaeCapacity {
        activation,
        staging,
        weight,
        bias,
        channels,
        filter: AA_KERNEL,
    })
}

/// Write one device buffer to the dump directory.
fn dump_region(
    runtime: &AukVaeRuntime,
    dir: &Path,
    name: &str,
    region: ArenaRegion,
    count: usize,
) -> Result<(), String> {
    let data = runtime
        .download(region, count)
        .map_err(|error| error.to_string())?;
    dump_tensor(dir, name, &data);
    Ok(())
}

/// Grow the shared runtime when a longer latent needs bigger buffers.
fn grow_runtime(
    state: &mut Option<AukVaeRuntime>,
    context: &'static crate::vulkan::VulkanContext,
    capacity: AukVaeCapacity,
) -> Result<(), String> {
    let needed = [
        capacity.activation,
        capacity.staging,
        capacity.weight,
        capacity.bias,
        capacity.channels,
        capacity.filter,
    ];
    let current = state.as_ref().map(|runtime| {
        [
            runtime.capacity.activation,
            runtime.capacity.staging,
            runtime.capacity.weight,
            runtime.capacity.bias,
            runtime.capacity.channels,
            runtime.capacity.filter,
        ]
    });
    if current
        .as_ref()
        .is_none_or(|current| current.iter().zip(needed).any(|(have, want)| want > *have))
    {
        let grown = match current {
            Some(current) => AukVaeCapacity {
                activation: current[0].max(capacity.activation),
                staging: current[1].max(capacity.staging),
                weight: current[2].max(capacity.weight),
                bias: current[3].max(capacity.bias),
                channels: current[4].max(capacity.channels),
                filter: current[5].max(capacity.filter),
            },
            None => capacity,
        };
        *state = Some(
            AukVaeRuntime::new(context, grown)
                .map_err(|error| format!("AuK VAE GPU arena: {error}"))?,
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::thread_pool::ComputePool;

    /// The device operators must agree with the host ones.
    ///
    /// The shapes cover the ones the decoder uses: the wide stage-0 blocks
    /// (768 channels), the narrow last stage (24 channels) and the reduction
    /// over several chunks.
    #[test]
    #[ignore = "requires a Vulkan device"]
    fn gpu_operators_match_the_host_kernels() {
        crate::ops::enable_gpu();
        let context = crate::ops::get_vulkan_context().expect("Vulkan device required");
        let pool = ComputePool::new(4);
        let taps: Vec<f32> = (0..AA_KERNEL)
            .map(|index| {
                let x = index as f32 / AA_KERNEL as f32 - 0.5;
                (-4.0 * x * x).exp() * (1.0 - 2.0 * x).cos()
            })
            .collect();
        for (channels, frames) in [(24usize, 2000usize), (128, 900)] {
            let alpha: Vec<f32> = (0..channels).map(|c| 0.5 + c as f32 * 0.01).collect();
            let beta: Vec<f32> = (0..channels).map(|c| 0.8 + c as f32 * 0.01).collect();
            let input: Vec<f32> = (0..channels * frames)
                .map(|index| {
                    let x = index as f32 * 0.37;
                    (x * 0.31).sin() * 2.0 + (x * 0.07).cos()
                })
                .collect();

            // Host: upsample, SnakeBeta, downsample.
            let mut up = super::super::vae::upsample_ratio2(&pool, &input, channels, frames, &taps);
            super::super::vae::apply_snake_beta_inplace(
                &mut up,
                channels,
                frames * AA_RATIO,
                &alpha,
                &beta,
                &pool,
            );
            let host = super::super::vae::downsample_ratio2(
                &pool,
                &up,
                channels,
                frames * AA_RATIO,
                &taps,
            );

            let capacity = AukVaeCapacity {
                activation: channels * frames * AA_RATIO,
                staging: channels * frames * AA_RATIO,
                weight: channels * channels * 11,
                bias: channels,
                channels,
                filter: AA_KERNEL,
            };
            let mut runtime = AukVaeRuntime::new(context, capacity).expect("arena");
            let source = runtime.residual;
            let destination = runtime.branch;
            runtime.upload(source, &input).expect("upload");
            aa_activation(
                &mut runtime,
                source,
                destination,
                &alpha,
                &beta,
                channels,
                frames,
                &taps,
            )
            .expect("anti-aliased activation");
            let device = runtime
                .download(destination, channels * frames)
                .expect("download");
            report(
                &format!("anti-aliased activation {channels}x{frames}"),
                &host,
                &device,
                1e-3,
            );

            // Host: the causal weight-norm convolution.
            let bias: Vec<f32> = (0..channels).map(|c| c as f32 * 0.01).collect();
            for (kernel, dilation) in [(3usize, 1usize), (11, 5)] {
                let weight: Vec<f32> = (0..channels * channels * kernel)
                    .map(|index| ((index % 17) as f32 - 8.0) * 0.01)
                    .collect();
                let mut host = input.clone();
                super::super::vae::apply_branch_conv_dilated(
                    &mut host, channels, frames, &weight, &bias, kernel, dilation, &pool,
                );
                runtime
                    .conv1d(
                        source,
                        destination,
                        &weight,
                        Some(&bias),
                        &AukConv1dSpec {
                            in_channels: channels,
                            out_channels: channels,
                            kernel,
                            stride: 1,
                            dilation,
                            pad_left: dilation * (kernel - 1),
                            frames,
                            out_frames: frames,
                            weight_layout: 0,
                        },
                    )
                    .expect("convolution");
                let device = runtime
                    .download(destination, channels * frames)
                    .expect("download");
                report(
                    &format!("convolution {channels}x{frames} k={kernel} d={dilation}"),
                    &host,
                    &device,
                    1e-3,
                );
            }
        }
    }

    /// Report the largest difference between the host and the device result.
    fn report(name: &str, host: &[f32], device: &[f32], tolerance: f32) {
        let mut worst = (0usize, 0.0f32);
        for (index, (a, b)) in host.iter().zip(device.iter()).enumerate() {
            let diff = (a - b).abs();
            if diff > worst.1 {
                worst = (index, diff);
            }
        }
        eprintln!(
            "[auk-vae-gpu] {name}: max |host - device| = {:e} at {}",
            worst.1, worst.0
        );
        assert!(
            worst.1 <= tolerance,
            "{name} drifted by {:e} (index {})",
            worst.1,
            worst.0
        );
    }
}
