use super::{math::*, weights::Graph, Options};
use crate::compute::linear::LinearExecutor;
use crate::core::thread_pool::ComputePool;
use crate::ops::gelu_erf_inplace;
use crate::ops::nn::{conv3d_with_options, Conv3dSpec};

#[allow(clippy::too_many_arguments)]
pub(super) fn decode<'a>(
    g: &'a Graph<'a>,
    ex: &mut LinearExecutor<'a, 'a>,
    pool: &ComputePool,
    features: &[f32],
    modulation: &[f32],
    options: &Options,
) -> Result<Vec<u8>, String> {
    const D: usize = 1536;
    let frames = options.latent_frames();
    let h = options.height / 16;
    let w = options.width / 16;
    let rows = frames * h * w;
    let mut x = linear(g, ex, "decoder.bridge", features)?;
    let m = linear(g, ex, "decoder.bridge", modulation)?;
    x = norm(&x, D, 1e-6)?;
    let e = &g.floats["decoder.pixel_head.modulation"];
    for (frame, x) in x.chunks_exact_mut(h * w * D).enumerate() {
        let shift = e[..D]
            .iter()
            .zip(&m[frame * D..(frame + 1) * D])
            .map(|(a, b)| a + b)
            .collect::<Vec<_>>();
        let scale = e[D..]
            .iter()
            .zip(&m[frame * D..(frame + 1) * D])
            .map(|(a, b)| a + b)
            .collect::<Vec<_>>();
        modulate(x, &shift, &scale);
    }
    x = affine_norm(g, "decoder.pixel_head.entry_norm", &x, 1e-6)?;
    x = linear(g, ex, "decoder.pixel_head.entry_pw", &x)?;
    for p in [
        "decoder.pixel_head.conv_low.0",
        "decoder.pixel_head.conv_low.1",
        "decoder.pixel_head.conv_high.0",
    ] {
        let mut z = causal_depthwise(
            pool,
            &x,
            D,
            frames,
            h,
            w,
            &g.floats[&format!("{p}.dw.weight")],
            &g.floats[&format!("{p}.dw.bias")],
        )?;
        z = affine_norm(g, &format!("{p}.norm"), &z, 1e-6)?;
        z = linear(g, ex, &format!("{p}.pw1"), &z)?;
        gelu_erf_inplace(&mut z);
        let z = linear(g, ex, &format!("{p}.pw2"), &z)?;
        residual(&mut x, &z, None);
        trace(p, &x);
    }
    assert_eq!(x.len(), rows * D);
    let pixels = linear(g, ex, "decoder.pixel_head.fc_out", &x)?;
    if pixels.iter().any(|v| !v.is_finite()) {
        return Err("non-finite FastFrames-L decoder pixels".into());
    }
    trace("fastframes.pixel_tokens", &pixels);
    Ok(unpatchify_pixels(&pixels, frames, h, w, options.frames))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn causal_depthwise(
    pool: &ComputePool,
    input: &[f32],
    channels: usize,
    frames: usize,
    h: usize,
    w: usize,
    weight: &[f32],
    bias: &[f32],
) -> Result<Vec<f32>, String> {
    let mut padded = vec![0.; channels * (frames + 2) * h * w];
    for f in 0..frames {
        for y in 0..h {
            for x in 0..w {
                for c in 0..channels {
                    padded[((c * (frames + 2) + f + 2) * h + y) * w + x] =
                        input[((f * h + y) * w + x) * channels + c];
                }
            }
        }
    }
    let (value, shape) = conv3d_with_options(
        pool,
        &padded,
        [channels, frames + 2, h, w],
        weight,
        [channels, 1, 3, 3, 3],
        Some(bias),
        Conv3dSpec {
            stride: [1, 1, 1],
            padding: [0, 1, 1],
            dilation: [1, 1, 1],
            groups: channels,
        },
    )?;
    if shape != [channels, frames, h, w] {
        return Err("FastFrames-L decoder convolution shape mismatch".into());
    }
    let mut output = vec![0.; input.len()];
    for (i, x) in output.iter_mut().enumerate() {
        let row = i / channels;
        let c = i % channels;
        *x = value[c * frames * h * w + row];
    }
    Ok(output)
}

pub(super) fn unpatchify_pixels(
    pixels: &[f32],
    frames: usize,
    h: usize,
    w: usize,
    requested: usize,
) -> Vec<u8> {
    let drop = frames * 4 - requested;
    let mut rgb = vec![0u8; requested * h * 16 * w * 16 * 3];
    for t in drop..frames * 4 {
        for y in 0..h * 16 {
            for x in 0..w * 16 {
                for c in 0..3 {
                    let src = (((((t / 4 * h + y / 16) * w + x / 16) * 4 + t % 4) * 16 + y % 16)
                        * 16
                        + x % 16)
                        * 3
                        + c;
                    let dst = (((t - drop) * h * 16 + y) * w * 16 + x) * 3 + c;
                    rgb[dst] = ((pixels[src].clamp(-1., 1.) + 1.) * 127.5).round_ties_even() as u8;
                }
            }
        }
    }
    rgb
}
