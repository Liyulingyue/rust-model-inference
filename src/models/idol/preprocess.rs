//! Official object crop and black compositing, including Pillow's 8-bit Lanczos rounding.
use image::{DynamicImage, GrayImage, RgbImage, RgbaImage};
use rayon::prelude::*;

pub(super) fn image(
    input: DynamicImage,
    image_size: u32,
    crop_margin: f64,
    background: impl FnOnce(&RgbImage) -> Result<GrayImage, String>,
) -> Result<RgbImage, String> {
    if input.width() == 0
        || input.height() == 0
        || image_size == 0
        || image_size > 1024
        || !crop_margin.is_finite()
        || crop_margin <= 0.
    {
        return Err("Invalid Idol preprocessing image size or crop margin".into());
    }
    let has_rgba = matches!(
        input,
        DynamicImage::ImageRgba8(_) | DynamicImage::ImageRgba16(_)
    );
    let scale = image_size as f64 / input.width().min(input.height()) as f64;
    let width = (input.width() as f64 * scale).round_ties_even().max(1.);
    let height = (input.height() as f64 * scale).round_ties_even().max(1.);
    if width > 32768. || height > 32768. {
        return Err("Idol input aspect ratio exceeds preprocessing allocation limit".into());
    }
    let mut rgba = resize_rgba(&input.to_rgba8(), width as u32, height as u32)?;
    if !has_rgba || rgba.pixels().all(|p| p[3] == 255) {
        let rgb = RgbImage::from_fn(rgba.width(), rgba.height(), |x, y| {
            let p = rgba.get_pixel(x, y);
            image::Rgb([p[0], p[1], p[2]])
        });
        let alpha = background(&rgb)?;
        if alpha.dimensions() != rgba.dimensions() {
            return Err("Idol background mask dimensions do not match the image".into());
        }
        for (pixel, mask) in rgba.pixels_mut().zip(alpha.pixels()) {
            pixel[3] = mask[0];
        }
    }
    let alpha: Vec<u8> = rgba.pixels().map(|p| p[3]).collect();
    let width = rgba.width() as usize;
    let height = rgba.height() as usize;
    let mut left = width;
    let mut right = 0;
    let mut top = height;
    let mut bottom = 0;
    for y in 0..height {
        for x in 0..width {
            let mut minimum = 255;
            for yy in y.saturating_sub(1)..=(y + 1).min(height - 1) {
                for xx in x.saturating_sub(1)..=(x + 1).min(width - 1) {
                    minimum = minimum.min(alpha[yy * width + xx]);
                }
            }
            rgba.get_pixel_mut(x as u32, y as u32)[3] = minimum;
            if minimum > 0 {
                left = left.min(x);
                right = right.max(x);
                top = top.min(y);
                bottom = bottom.max(y);
            }
        }
    }
    if left == width {
        return Err("No object found in the image (the alpha channel is empty)".into());
    }
    let center_x = (left + right) as f64 / 2.;
    let center_y = (top + bottom) as f64 / 2.;
    let half = (right - left).max(bottom - top) as f64 / 2. * crop_margin;
    if !half.is_finite() || half > 32768. {
        return Err("Idol alpha crop exceeds preprocessing allocation limit".into());
    }
    let x0 = (center_x - half).trunc() as i64;
    let x1 = (center_x + half).trunc() as i64;
    let y0 = (center_y - half).trunc() as i64;
    let y1 = (center_y + half).trunc() as i64;
    if x1 <= x0 || y1 <= y0 || x1 - x0 > 32768 || y1 - y0 > 32768 {
        return Err("Idol alpha bounding box has an invalid crop size".into());
    }
    if (x1 - x0) * (y1 - y0) > 64 * 1024 * 1024 {
        return Err("Idol alpha crop exceeds preprocessing allocation limit".into());
    }
    let cropped = RgbaImage::from_fn((x1 - x0) as u32, (y1 - y0) as u32, |x, y| {
        let xx = x0 + x as i64;
        let yy = y0 + y as i64;
        if xx < 0 || yy < 0 || xx >= width as i64 || yy >= height as i64 {
            image::Rgba([0; 4])
        } else {
            *rgba.get_pixel(xx as u32, yy as u32)
        }
    });
    let rgba = resize_rgba(&cropped, image_size, image_size)?;
    Ok(RgbImage::from_fn(image_size, image_size, |x, y| {
        let p = rgba.get_pixel(x, y);
        image::Rgb([mul255(p[0], p[3]), mul255(p[1], p[3]), mul255(p[2], p[3])])
    }))
}
fn mul255(x: u8, a: u8) -> u8 {
    let value = x as u32 * a as u32 + 128;
    (((value >> 8) + value) >> 8) as u8
}
fn coefficients(input: usize, output: usize) -> Vec<(usize, Vec<i32>)> {
    let scale = input as f64 / output as f64;
    let filter_scale = scale.max(1.);
    let support = 3. * filter_scale;
    (0..output)
        .map(|p| {
            let center = (p as f64 + 0.5) * scale;
            let start = ((center - support + 0.5) as isize).max(0) as usize;
            let end = ((center + support + 0.5) as usize).min(input);
            let weights: Vec<_> = (start..end)
                .map(|i| {
                    let x = (i as f64 - center + 0.5) / filter_scale;
                    if x == 0. {
                        1.
                    } else if !(-3.0..3.0).contains(&x) {
                        0.
                    } else {
                        let t = x * std::f64::consts::PI;
                        (t.sin() / t) * ((t / 3.).sin() / (t / 3.))
                    }
                })
                .collect();
            let sum: f64 = weights.iter().sum();
            (
                start,
                weights
                    .into_iter()
                    .map(|w| {
                        let w = w / sum * (1u32 << 22) as f64;
                        (w + if w < 0. { -0.5 } else { 0.5 }) as i32
                    })
                    .collect(),
            )
        })
        .collect()
}
fn resize_rgba(input: &RgbaImage, width: u32, height: u32) -> Result<RgbaImage, String> {
    if input.dimensions() == (width, height) {
        return Ok(input.clone());
    }
    let mut premultiplied = input.clone();
    for p in premultiplied.pixels_mut() {
        for c in 0..3 {
            p[c] = mul255(p[c], p[3]);
        }
    }
    let iw = input.width() as usize;
    let ih = input.height() as usize;
    let width = width as usize;
    let height = height as usize;
    let hcoeff = coefficients(iw, width);
    let vcoeff = coefficients(ih, height);
    let mut horizontal = vec![
        0u8;
        width
            .checked_mul(ih)
            .and_then(|n| n.checked_mul(4))
            .ok_or("Idol image resize overflow")?
    ];
    horizontal
        .par_chunks_mut(width * 4)
        .enumerate()
        .for_each(|(y, row)| {
            for (x, (start, weights)) in hcoeff.iter().enumerate() {
                for c in 0..4 {
                    let mut sum = 1i64 << 21;
                    for (i, &weight) in weights.iter().enumerate() {
                        sum += premultiplied.as_raw()[(y * iw + start + i) * 4 + c] as i64
                            * weight as i64;
                    }
                    row[x * 4 + c] = (sum >> 22).clamp(0, 255) as u8;
                }
            }
        });
    let mut output = vec![
        0u8;
        width
            .checked_mul(height)
            .and_then(|n| n.checked_mul(4))
            .ok_or("Idol image resize overflow")?
    ];
    output
        .par_chunks_mut(width * 4)
        .enumerate()
        .for_each(|(y, row)| {
            let (start, weights) = &vcoeff[y];
            for x in 0..width {
                for c in 0..4 {
                    let mut sum = 1i64 << 21;
                    for (i, &weight) in weights.iter().enumerate() {
                        sum += horizontal[((start + i) * width + x) * 4 + c] as i64 * weight as i64;
                    }
                    row[x * 4 + c] = (sum >> 22).clamp(0, 255) as u8;
                }
            }
            for p in row.chunks_exact_mut(4) {
                if p[3] != 0 && p[3] != 255 {
                    for c in 0..3 {
                        p[c] = (255 * p[c] as u32 / p[3] as u32).min(255) as u8;
                    }
                }
            }
        });
    RgbaImage::from_raw(width as u32, height as u32, output)
        .ok_or_else(|| "Invalid Idol resized image storage".into())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn transparent_input_skips_remover_and_empty_alpha_fails() {
        let rgba = RgbaImage::from_fn(16, 16, |x, y| {
            image::Rgba([
                255,
                128,
                64,
                if (3..13).contains(&x) && (3..13).contains(&y) {
                    255
                } else {
                    0
                },
            ])
        });
        let output = image(DynamicImage::ImageRgba8(rgba), 16, 1.2, |_| {
            panic!("alpha must bypass background remover")
        })
        .unwrap();
        assert_eq!(output.dimensions(), (16, 16));
        assert!(output.pixels().any(|p| p[0] > 0));
        assert!(image(
            DynamicImage::ImageRgba8(RgbaImage::new(16, 16)),
            16,
            1.2,
            |_| panic!()
        )
        .is_err());
    }
    #[test]
    fn opaque_input_requires_mask_and_lanczos_keeps_constant() {
        let rgba = RgbaImage::from_pixel(7, 9, image::Rgba([17, 33, 65, 255]));
        let output = resize_rgba(&rgba, 11, 13).unwrap();
        assert!(output.pixels().all(|p| p.0 == [17, 33, 65, 255]));
        let error = image(DynamicImage::ImageRgba8(rgba), 16, 1.2, |_| {
            Err("remover called".into())
        })
        .unwrap_err();
        assert_eq!(error, "remover called");
    }
}
