//! Test images shared by the image-input and protocol tests.
//!
//! Originally this embedded a copy of `references/apple.png` (the image
//! `docs/usage/qwen3.md` uses). That tree is gitignored — vendored
//! third-party reference material — so committing a copy of one of its
//! binaries into `tests/` was inconsistent with the repo's hygiene. The
//! image is now generated at test time instead: same 401x287 dimensions
//! (and therefore the same vision-token count), deterministic content,
//! zero binary assets.

/// Width of the synthetic test image. Chosen to match the docs image so the
/// tests exercise the same resize / grid path.
pub(crate) const IMAGE_WIDTH: u32 = 401;
pub(crate) const IMAGE_HEIGHT: u32 = 287;

/// A deterministic RGB PNG (`IMAGE_WIDTH` x `IMAGE_HEIGHT`).
pub(crate) fn synthetic_png() -> Vec<u8> {
    use image::{ImageFormat, Rgb, RgbImage};
    use std::io::Cursor;

    let mut image = RgbImage::new(IMAGE_WIDTH, IMAGE_HEIGHT);
    for (x, y, pixel) in image.enumerate_pixels_mut() {
        *pixel = Rgb([x as u8, y as u8, (x ^ y) as u8]);
    }
    let mut buffer = Cursor::new(Vec::new());
    image::DynamicImage::ImageRgb8(image)
        .write_to(&mut buffer, ImageFormat::Png)
        .expect("synthetic PNG encode");
    buffer.into_inner()
}

/// Base64 of [`synthetic_png`], for the data-URI / raw-base64 test shapes.
pub(crate) fn synthetic_png_b64() -> String {
    use base64::Engine as _;
    base64::engine::general_purpose::STANDARD.encode(synthetic_png())
}
