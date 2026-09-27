//! Test fixtures shared by the image-input and protocol tests.
//!
//! `tests/fixtures/apple.png` is the same image the docs use
//! (`docs/usage/qwen3.md --image references/apple.png`). It is embedded from
//! the manifest dir (not a relative path) so it does not matter which module
//! includes it, and it lives under `tests/fixtures/` rather than
//! `references/` because `references/` is gitignored — CI must be able to
//! read it.

/// 401x287 8-bit RGB PNG, ~103 KB — a realistic payload rather than a stub.
pub(crate) const APPLE_PNG: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/apple.png"
));

pub(crate) const APPLE_PNG_WIDTH: u32 = 401;
pub(crate) const APPLE_PNG_HEIGHT: u32 = 287;

/// Base64 of [`APPLE_PNG`], for the data-URI / raw-base64 test shapes.
pub(crate) fn apple_png_b64() -> String {
    use base64::Engine as _;
    base64::engine::general_purpose::STANDARD.encode(APPLE_PNG)
}
