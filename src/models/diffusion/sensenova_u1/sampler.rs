//! Flow-matching sampler for SenseNova-U1.6 (pixel-space, no VAE).
//!
//! Replays `NEOChatModel.t2i_generate` from the pinned oracle
//! (`OpenSenseNova/SenseNova-U1` @ `d2d9bd1`). Every numeric constant here is
//! taken from that function, not from a re-derivation; see
//! `docs/develop/SENSENOVA_U1.md` §5.

use crate::core::thread_pool::ComputePool;

use super::config;
use super::fm::{ConvDecoder, TimestepEmbedder};
use super::trunk::{PrefixCache, Trunk};
use super::vision::VisionEmbeddings;

/// Sampler knobs. Defaults mirror `examples/t2i/inference.py`.
#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub width: usize,
    pub height: usize,
    pub steps: usize,
    pub cfg_scale: f32,
    pub timestep_shift: f32,
    pub seed: u64,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            width: 1024,
            height: 1024,
            steps: config::DEFAULT_STEPS,
            cfg_scale: config::DEFAULT_CFG_SCALE,
            timestep_shift: config::DEFAULT_TIMESTEP_SHIFT,
            seed: 42,
        }
    }
}

/// Everything one denoise loop needs, loaded once.
pub(crate) struct Sampler {
    pub(crate) vision_gen: VisionEmbeddings,
    pub(crate) timestep: TimestepEmbedder,
    pub(crate) noise_scale_embedder: TimestepEmbedder,
    pub(crate) decoder: ConvDecoder,
    pub(crate) trunk: Trunk,
}

/// Reject geometry the pixel-space head cannot represent.
///
/// The image token grid is `H / (patch * merge)` and the ConvDecoder grows it
/// back by `2 * 2 * 8 = 32` per side, so both sides must be a multiple of 32
/// and the token grid must be non-empty.
pub(crate) fn validate_geometry(width: usize, height: usize) -> Result<(), String> {
    for (label, value) in [("width", width), ("height", height)] {
        if value == 0 || !value.is_multiple_of(config::TOKEN_SIDE) {
            return Err(format!(
                "SenseNova-U1.6 {label} {value} must be a positive multiple of {}",
                config::TOKEN_SIDE
            ));
        }
    }
    Ok(())
}

/// `torch.linspace(0, 1, steps + 1)` passed through the flow-time shift.
///
/// Upstream `_apply_time_schedule` hardcodes `time_schedule = "standard"`, so
/// the `dynamic` / `base_shift` / `max_shift` branch in `config.json` is dead
/// code for this release and is deliberately not implemented.
pub(crate) fn timesteps(steps: usize, shift: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(steps + 1);
    for index in 0..=steps {
        let t = index as f32 / steps as f32;
        let sigma = 1.0 - t;
        let sigma = if shift != 1.0 {
            shift * sigma / (1.0 + (shift - 1.0) * sigma)
        } else {
            sigma
        };
        out.push(1.0 - sigma);
    }
    out
}

/// `noise_scale` for a given image size (`noise_scale_mode = "resolution"`).
///
/// `scale = sqrt((grid_h * grid_w) / merge^2 / base_image_seq_len)`, clamped to
/// `noise_scale_max_value`. For 512x512 that is `sqrt(1024/4/64) = 2.0`.
pub(crate) fn noise_scale(width: usize, height: usize) -> f32 {
    let grid_h = height / config::PATCH_SIZE;
    let grid_w = width / config::PATCH_SIZE;
    let base = config::NOISE_SCALE_BASE_IMAGE_SEQ_LEN;
    let scale =
        ((grid_h * grid_w) as f64 / (config::MERGE_SIZE * config::MERGE_SIZE) as f64 / base).sqrt();
    (scale as f32 * config::NOISE_SCALE).min(config::NOISE_SCALE_MAX)
}

/// `patchify(images, patch, channel_first=False)` from the oracle.
///
/// `images` is `[3, h, w]` channel-major; the result is `[tokens, patch*patch*3]`
/// with the inner order `(p, q, c)` — row inside the patch, then column, then
/// channel.
pub(crate) fn patchify(images: &[f32], width: usize, height: usize, patch: usize) -> Vec<f32> {
    let h = height / patch;
    let w = width / patch;
    let mut out = vec![0.0; h * w * patch * patch * 3];
    for py in 0..h {
        for px in 0..w {
            let token = py * w + px;
            for p in 0..patch {
                for q in 0..patch {
                    for c in 0..3 {
                        let src = (c * height + py * patch + p) * width + px * patch + q;
                        let dst = token * patch * patch * 3 + (p * patch + q) * 3 + c;
                        out[dst] = images[src];
                    }
                }
            }
        }
    }
    out
}

/// `patchify(images, patch, channel_first=True)`: inner order `(c, p, q)`.
pub(crate) fn patchify_channel_first(
    images: &[f32],
    width: usize,
    height: usize,
    patch: usize,
) -> Vec<f32> {
    let h = height / patch;
    let w = width / patch;
    let mut out = vec![0.0; h * w * patch * patch * 3];
    for py in 0..h {
        for px in 0..w {
            let token = py * w + px;
            for p in 0..patch {
                for q in 0..patch {
                    for c in 0..3 {
                        let src = (c * height + py * patch + p) * width + px * patch + q;
                        let dst = token * patch * patch * 3 + (c * patch + p) * patch + q;
                        out[dst] = images[src];
                    }
                }
            }
        }
    }
    out
}

/// `unpatchify(x, patch, h, w)`: inverse of [`patchify`].
pub(crate) fn unpatchify(tokens: &[f32], width: usize, height: usize, patch: usize) -> Vec<f32> {
    let h = height / patch;
    let w = width / patch;
    let mut out = vec![0.0; 3 * height * width];
    for py in 0..h {
        for px in 0..w {
            let token = py * w + px;
            for p in 0..patch {
                for q in 0..patch {
                    for c in 0..3 {
                        let src = token * patch * patch * 3 + (p * patch + q) * 3 + c;
                        let dst = (c * height + py * patch + p) * width + px * patch + q;
                        out[dst] = tokens[src];
                    }
                }
            }
        }
    }
    out
}

/// Whether checkpoint tracing is on.
///
/// Callers guard their `format!` with this so the default (untraced) path never
/// allocates a checkpoint name inside the denoise loop.
#[inline]
pub(crate) fn tracing() -> bool {
    std::env::var_os("RMI_SENSENOVA_TRACE").is_some()
}

/// Optional checkpoint dump for oracle alignment.
///
/// Enabled by `RMI_SENSENOVA_TRACE=<dir>`. Each call writes `<dir>/<name>.f32`
/// (raw little-endian f32) plus a one-line `<dir>/trace.log` entry with the
/// shape, so a Python script can `np.fromfile(..., dtype="<f4")` the same
/// tensors the oracle dumps as `.npy`. Off by default: the only cost when the
/// variable is unset is one `env::var_os` per checkpoint.
pub(crate) fn trace_dump(name: &str, values: &[f32]) {
    let Some(dir) = std::env::var_os("RMI_SENSENOVA_TRACE") else {
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let _ = std::fs::create_dir_all(&dir);
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    let _ = std::fs::write(dir.join(format!("{name}.f32")), &bytes);
    use std::io::Write as _;
    if let Ok(mut log) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join("trace.log"))
    {
        let _ = writeln!(log, "{name} {}", values.len());
    }
}

/// Scratch buffers reused across the whole denoise loop.
struct Buffers {
    /// Current noisy image `[3, H, W]` in normalised space.
    image: Vec<f32>,
    /// `patchify(image, 32)` — the flow target `z` `[tokens, 3072]`.
    z: Vec<f32>,
    /// `patchify(image, 16, channel_first=True)` `[grid*grid, 768]`.
    patches: Vec<f32>,
    /// Vision embeddings + timestep/noise embeddings `[tokens, 4096]`.
    embeds: Vec<f32>,
    v_cond: Vec<f32>,
    v_uncond: Vec<f32>,
    /// ConvDecoder input, `[4096, token_h, token_w]`.
    decoder_in: Vec<f32>,
}

/// Run the full T2I loop and return the final image as `[3, H, W]` in `[0, 1]`.
///
/// `cache_cond` / `cache_uncond` are the prefix passes for the two CFG branches.
#[allow(clippy::too_many_arguments)]
pub(crate) fn denoise(
    sampler: &Sampler,
    source: &dyn crate::core::tensor::TensorSource,
    cache_cond: &PrefixCache,
    cache_uncond: Option<&PrefixCache>,
    text_len_cond: usize,
    text_len_uncond: usize,
    options: &Options,
    pool: &ComputePool,
) -> Result<Vec<f32>, String> {
    validate_geometry(options.width, options.height)?;
    if options.steps == 0 {
        return Err("SenseNova-U1.6 needs at least one denoise step".into());
    }

    let width = options.width;
    let height = options.height;
    let token_h = height / config::TOKEN_SIDE;
    let token_w = width / config::TOKEN_SIDE;
    let tokens = token_h * token_w;
    let grid_h = height / config::PATCH_SIZE;
    let grid_w = width / config::PATCH_SIZE;

    let noise_scale = noise_scale(width, height);
    let schedule = timesteps(options.steps, options.timestep_shift);

    // `torch.randn(..., generator=torch.Generator('cpu').manual_seed(seed))`.
    let mut buffers = Buffers {
        image: vec![0.0; 3 * height * width],
        z: Vec::new(),
        patches: Vec::new(),
        embeds: vec![0.0; tokens * config::HIDDEN],
        v_cond: vec![0.0; tokens * config::TOKEN_DIM],
        v_uncond: vec![0.0; tokens * config::TOKEN_DIM],
        decoder_in: vec![0.0; config::HIDDEN * tokens],
    };
    crate::models::diffusion::z_image::dit::TorchMt19937::new(options.seed)
        .fill_normal(&mut buffers.image);
    for value in &mut buffers.image {
        *value *= noise_scale;
    }

    // Constant per-token embeddings of the flow time and the noise scale.
    let noise_embedding = if sampler.noise_scale_embedder_is_active() {
        Some(
            sampler
                .noise_scale_embedder
                .forward(noise_scale / config::NOISE_SCALE_MAX),
        )
    } else {
        None
    };

    trace_dump("noise_init", &buffers.image);
    trace_dump("timesteps", &schedule);

    let needs_cfg = options.cfg_scale > 1.0;
    if needs_cfg && cache_uncond.is_none() {
        return Err("SenseNova-U1.6 CFG requires the unconditional prefix cache".into());
    }

    for step in 0..options.steps {
        let t = schedule[step];
        let t_next = schedule[step + 1];

        buffers.z = patchify(&buffers.image, width, height, config::TOKEN_SIDE);
        if tracing() {
            if step == 0 {
                trace_dump("step0_t", &[t]);
                trace_dump("step0_t_next", &[t_next]);
            }
            let name = format!("step{step}_z");
            trace_dump(&name, &buffers.z);
        }
        buffers.patches = patchify_channel_first(&buffers.image, width, height, config::PATCH_SIZE);

        let mut embeds = sampler
            .vision_gen
            .forward(&buffers.patches, grid_h, grid_w, pool)?;
        if embeds.len() != tokens * config::HIDDEN {
            return Err(format!(
                "SenseNova-U1.6 vision embeddings produced {} values, expected {}",
                embeds.len(),
                tokens * config::HIDDEN
            ));
        }

        let timestep = sampler.timestep.forward(t);
        let mut delta = vec![0.0; config::HIDDEN];
        for index in 0..config::HIDDEN {
            delta[index] =
                timestep[index] + noise_embedding.as_ref().map_or(0.0, |noise| noise[index]);
        }
        for token in 0..tokens {
            let row = &mut embeds[token * config::HIDDEN..(token + 1) * config::HIDDEN];
            for index in 0..config::HIDDEN {
                row[index] += delta[index];
            }
        }
        if step == 0 {
            trace_dump("step0_image_embeds", &embeds);
        }
        buffers.embeds = embeds;

        let mut predict = |cache: &PrefixCache,
                           text_len: usize,
                           out: &mut Vec<f32>,
                           branch: &str|
         -> Result<(), String> {
            let hidden =
                sampler
                    .trunk
                    .denoise(source, &buffers.embeds, text_len, token_w, cache, pool)?;
            // `img_reshaped = hidden[:, -L:].view(B, token_h, token_w, -1)` then
            // `einsum("b h w c -> b c h w")`.
            let base = hidden.len() - tokens * config::HIDDEN;
            for token in 0..tokens {
                let h = token / token_w;
                let w = token % token_w;
                for c in 0..config::HIDDEN {
                    buffers.decoder_in[(c * token_h + h) * token_w + w] =
                        hidden[base + token * config::HIDDEN + c];
                }
            }
            if step == 0 && tracing() {
                trace_dump(&format!("step0_decoder_in_{branch}"), &buffers.decoder_in);
                trace_dump(&format!("step0_hidden_{branch}"), &hidden);
                // Branch-explicit copies so the decoder stages pair with the
                // right input: the per-call `dec_*` files below are overwritten
                // by the CFG-unconditional pass, so they can only be compared
                // against `..._uncond`.
                super::fm::set_trace_suffix(branch);
            }
            let decoded = sampler
                .decoder
                .forward(&buffers.decoder_in, token_h, token_w, pool)?;
            // `view(B, 3, token_h, 32, token_w, 32)` then
            // `einsum("b c h p w q -> b h w p q c")`.
            out.clear();
            out.resize(tokens * config::TOKEN_DIM, 0.0);
            for c in 0..3 {
                for h in 0..token_h {
                    for p in 0..config::PATCH_SIZE {
                        for w in 0..token_w {
                            for q in 0..config::PATCH_SIZE {
                                let src = ((c * token_h + h) * config::TOKEN_SIDE + p) * width
                                    + w * config::TOKEN_SIDE
                                    + q;
                                let token = h * token_w + w;
                                let dst = token * config::TOKEN_DIM
                                    + ((p * config::PATCH_SIZE + q) * 3 + c);
                                out[dst] = decoded[src];
                            }
                        }
                    }
                }
            }
            Ok(())
        };

        if needs_cfg {
            predict(cache_cond, text_len_cond, &mut buffers.v_cond, "cond")?;
            predict(
                cache_uncond.expect("checked above"),
                text_len_uncond,
                &mut buffers.v_uncond,
                "uncond",
            )?;
        } else {
            predict(cache_cond, text_len_cond, &mut buffers.v_cond, "cond")?;
        }

        // The head predicts `x_pred` in sample space; the flow velocity is
        // `v = (x_pred - z) / clamp(1 - t, min = t_eps)`. Skipping this step
        // leaves `z` parked on the initial noise -- which is exactly the
        // "pure noise output" failure mode this port hit first.
        let denominator = (1.0 - t).max(config::DEFAULT_T_EPS);
        let scale = options.cfg_scale;
        for index in 0..tokens * config::TOKEN_DIM {
            let cond = (buffers.v_cond[index] - buffers.z[index]) / denominator;
            let velocity = if needs_cfg {
                let uncond = (buffers.v_uncond[index] - buffers.z[index]) / denominator;
                uncond + scale * (cond - uncond)
            } else {
                cond
            };
            buffers.z[index] += (t_next - t) * velocity;
        }

        if step == 0 {
            // Trace the post-division velocities so they pair with the
            // oracle's `step*_v_pred_cond` / `step*_v_pred` dumps.
            let denom = denominator;
            let mut cond_v = vec![0.0; tokens * config::TOKEN_DIM];
            let mut uncond_v = vec![0.0; tokens * config::TOKEN_DIM];
            for index in 0..tokens * config::TOKEN_DIM {
                cond_v[index] = (buffers.v_cond[index] - buffers.z[index]) / denom;
                uncond_v[index] = (buffers.v_uncond[index] - buffers.z[index]) / denom;
            }
            trace_dump("step0_v_pred_cond", &cond_v);
            trace_dump("step0_v_pred_uncond", &uncond_v);
            let mut combined = vec![0.0; tokens * config::TOKEN_DIM];
            for index in 0..tokens * config::TOKEN_DIM {
                combined[index] = if needs_cfg {
                    uncond_v[index] + scale * (cond_v[index] - uncond_v[index])
                } else {
                    cond_v[index]
                };
            }
            trace_dump("step0_v_pred", &combined);
            trace_dump("step0_x_pred_cond", &buffers.v_cond);
        }

        buffers.image = unpatchify(&buffers.z, width, height, config::TOKEN_SIDE);
        {
            let n = format!("step{step}_z_next");
            trace_dump(&n, &buffers.z);
            let n = format!("step{step}_image");
            trace_dump(&n, &buffers.image);
        }
    }

    trace_dump("final_image_prediction", &buffers.image);
    // `image_prediction` is returned in normalised space; the caller denormalises.
    Ok(buffers.image)
}

impl Sampler {
    /// Whether the checkpoint carries a noise-scale embedder.
    ///
    /// `add_noise_scale_embedding` is true for this release; the flag lives in
    /// `config.json` and the tensor is validated at load time, so the presence
    /// of the weights is the authority here.
    fn noise_scale_embedder_is_active(&self) -> bool {
        self.noise_scale_embedder.is_present()
    }
}

/// Denormalise `(x * 0.5 + 0.5)` and quantise to 8-bit RGB, matching
/// `_denorm` + `_to_pil` in the oracle.
pub(crate) fn to_rgb8(image: &[f32], width: usize, height: usize) -> Vec<u8> {
    let mut out = vec![0u8; width * height * 3];
    for (index, value) in image.iter().enumerate() {
        let normalised = (value * 0.5 + 0.5).clamp(0.0, 1.0);
        out[index] = (normalised * 255.0).round() as u8;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timesteps_match_the_oracle_shift() {
        // 4 steps, shift 3.0 -> t = [0.0, 0.1, 0.25, 0.5, 1.0]
        let schedule = timesteps(4, 3.0);
        let expected = [0.0, 0.1, 0.25, 0.5, 1.0];
        assert_eq!(schedule.len(), expected.len());
        for (got, want) in schedule.iter().zip(expected) {
            assert!((got - want).abs() < 1e-6, "got {got}, want {want}");
        }
    }

    #[test]
    fn timesteps_identity_when_shift_is_one() {
        let schedule = timesteps(4, 1.0);
        for (index, value) in schedule.iter().enumerate() {
            assert!((value - index as f32 / 4.0).abs() < 1e-6);
        }
    }

    #[test]
    fn noise_scale_uses_merge_size_squared() {
        // sqrt((32*32) / 4 / 64) = sqrt(4) = 2.0
        assert!((noise_scale(512, 512) - 2.0).abs() < 1e-6);
        // sqrt((64*64) / 4 / 64) = sqrt(16) = 4.0
        assert!((noise_scale(1024, 1024) - 4.0).abs() < 1e-6);
    }

    #[test]
    fn patchify_unpatchify_round_trip() {
        let width = 64;
        let height = 32;
        let image: Vec<f32> = (0..3 * width * height).map(|v| v as f32).collect();
        for patch in [config::PATCH_SIZE, config::TOKEN_SIDE] {
            let tokens = patchify(&image, width, height, patch);
            assert_eq!(
                tokens.len(),
                (width / patch) * (height / patch) * patch * patch * 3
            );
            let restored = unpatchify(&tokens, width, height, patch);
            assert_eq!(restored, image, "round trip failed for patch {patch}");
        }
    }

    #[test]
    fn patchify_channel_first_orders_channels_inside_the_patch() {
        // One 16x16 patch, only channel 1 set at (p=2, q=3).
        let width = 16;
        let height = 16;
        let mut image = vec![0.0; 3 * width * height];
        image[(1 * height + 2) * width + 3] = 1.0;
        let tokens = patchify_channel_first(&image, width, height, 16);
        assert_eq!(tokens.len(), 3 * 16 * 16);
        let expected = (1 * 16 + 2) * 16 + 3;
        assert_eq!(tokens[expected], 1.0);
        assert_eq!(tokens.iter().sum::<f32>(), 1.0);

        // The non-channel-first layout puts (p, q, c) inside the patch.
        let tokens = patchify(&image, width, height, 16);
        let expected = ((2 * 16 + 3) * 3) + 1;
        assert_eq!(tokens[expected], 1.0);
        assert_eq!(tokens.iter().sum::<f32>(), 1.0);
    }

    #[test]
    fn geometry_validation_rejects_non_multiples() {
        assert!(validate_geometry(512, 512).is_ok());
        assert!(validate_geometry(1024, 768).is_ok());
        assert!(validate_geometry(500, 512).is_err());
        assert!(validate_geometry(0, 512).is_err());
    }

    #[test]
    fn to_rgb8_denormalises_and_clamps() {
        let image = vec![-1.0, 0.0, 1.0, 2.0];
        let rgb = to_rgb8(&image, 2, 2);
        // 4 values is only one third of a 2x2 RGB image, so only the first
        // three channels of the buffer are written by index.
        assert_eq!(rgb[0], 0);
        assert_eq!(rgb[1], 128);
        assert_eq!(rgb[2], 255);
    }
}
