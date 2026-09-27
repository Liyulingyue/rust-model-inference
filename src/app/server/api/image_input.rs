//! Image input decoding for the OpenAI-compatible surface.
//!
//! Accepts the shapes the OpenAI SDK, the Responses API and the Anthropic
//! Messages API actually send, and normalises them to raw bytes:
//!
//! * `data:image/png;base64,...` — how the SDK passes a local file
//! * raw base64 — how the Anthropic Messages API passes images
//! * `file:///path` or a bare local path — same-host convenience
//! * `http(s)://...` — **rejected** unless the server is started with
//!   `--allow-remote-images`; fetching remote URLs from an inference server
//!   is an SSRF surface (internal-network probing, request amplification),
//!   so it stays opt-in.
//!
//! The decode pattern mirrors what `/v1/audio/speech` already does for
//! `voice` data URIs (see `server/mod.rs`), and the bytes are handed to
//! `image::load_from_memory` downstream (see `app::media::decode_image`).
use base64::Engine as _;
use serde::{Deserialize, Serialize};

/// One decoded image plus the source string it came from (for errors).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageRef {
    /// Raw encoded bytes (PNG / JPEG / …), ready for `image::load_from_memory`.
    pub bytes: Vec<u8>,
    /// The original URL / data URI / path, kept for diagnostics.
    pub source: String,
}

impl ImageRef {
    pub fn new(bytes: Vec<u8>, source: impl Into<String>) -> Self {
        Self {
            bytes,
            source: source.into(),
        }
    }
}

/// Decode one image source string into bytes.
///
/// `allow_remote` is the server's `--allow-remote-images` flag; when false,
/// `http(s)` URLs are rejected before any network access is attempted.
pub fn decode_image_source(source: &str, allow_remote: bool) -> Result<ImageRef, String> {
    let trimmed = source.trim();
    if trimmed.is_empty() {
        return Err("image source is empty".into());
    }
    if let Some(rest) = strip_data_uri(trimmed) {
        let bytes = decode_base64(rest, "image data URI")?;
        return Ok(ImageRef::new(bytes, trimmed));
    }
    if trimmed.starts_with("http://") || trimmed.starts_with("https://") {
        if !allow_remote {
            return Err(
                "remote image URLs are disabled; start the server with --allow-remote-images \
                 to enable (or pass a data: URI / local file)"
                    .into(),
            );
        }
        return Err(
            "remote image fetching is not implemented yet; --allow-remote-images is not \
                    wired up in this build"
                .into(),
        );
    }
    if let Some(path) = trimmed.strip_prefix("file://") {
        return read_local_image(trimmed, path);
    }
    // Anything else: try raw base64 first (the Anthropic Messages shape sends
    // bare base64 with no data: prefix), then fall back to a local path.
    if looks_like_base64(trimmed) {
        if let Ok(bytes) = decode_base64(trimmed, "image") {
            return Ok(ImageRef::new(bytes, trimmed));
        }
    }
    read_local_image(trimmed, trimmed)
}

/// Pull the payload out of a `data:<mediatype>;base64,<payload>` URI.
fn strip_data_uri(source: &str) -> Option<&str> {
    let rest = source.strip_prefix("data:")?;
    let comma = rest.find(',')?;
    let meta = &rest[..comma];
    if !meta.contains("base64") {
        return None;
    }
    Some(&rest[comma + 1..])
}

fn decode_base64(payload: &str, what: &str) -> Result<Vec<u8>, String> {
    let cleaned: String = payload.chars().filter(|c| !c.is_whitespace()).collect();
    base64::engine::general_purpose::STANDARD
        .decode(cleaned.as_bytes())
        .map_err(|error| format!("{what} base64 decode failed: {error}"))
}

/// Cheap pre-check so a local path like `images/cat.png` is not mistaken for
/// base64. The base64 alphabet is `A-Za-z0-9+/=`, so a `.` (file extension)
/// or a `\` (Windows separator) rules it out; `/` does NOT, because base64
/// uses it.
fn looks_like_base64(source: &str) -> bool {
    source.len() > 32
        && !source.contains('.')
        && !source.contains('\\')
        && source
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '+' || c == '/' || c == '=')
}

fn read_local_image(source: &str, path: &str) -> Result<ImageRef, String> {
    let bytes = std::fs::read(path)
        .map_err(|error| format!("failed to read image file {path:?}: {error}"))?;
    Ok(ImageRef::new(bytes, source))
}

/// Decode bytes into a `DynamicImage`, mirroring `app::media::decode_image`
/// but for already-in-memory bytes (base64 / uploads).
pub fn decode_image_bytes(bytes: &[u8]) -> Result<image::DynamicImage, String> {
    image::load_from_memory(bytes).map_err(|error| format!("failed to decode image: {error}"))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::app::server::api::fixtures::{apple_png_b64, APPLE_PNG};

    /// The apple.png fixture the docs use, embedded so tests do not depend on
    /// the gitignored `references/` tree. 401x287 RGB.
    use serde::{Deserialize, Serialize};

    #[test]
    fn data_uri_decodes_to_png_bytes() {
        let uri = format!("data:image/png;base64,{}", apple_png_b64());
        let image = decode_image_source(&uri, false).unwrap();
        // Assert on the decoded image, not a byte count: 1x1 transparent PNG.
        let decoded = decode_image_bytes(&image.bytes).unwrap();
        assert_eq!((decoded.width(), decoded.height()), (401, 287));
    }

    #[test]
    fn raw_base64_decodes_without_data_prefix() {
        let b64 = apple_png_b64();
        let image = decode_image_source(&b64, false).unwrap();
        assert_eq!(decode_image_bytes(&image.bytes).unwrap().width(), 401);
    }

    #[test]
    fn remote_url_is_rejected_by_default_with_actionable_message() {
        let error = decode_image_source("https://example.com/cat.png", false).unwrap_err();
        assert!(
            error.contains("--allow-remote-images"),
            "message must say how to enable it: {error}"
        );
    }

    #[test]
    fn remote_url_still_rejected_when_flag_off_but_explicit_http() {
        let error = decode_image_source("http://10.0.0.1/logo.png", false).unwrap_err();
        assert!(error.contains("remote image URLs are disabled"), "{error}");
    }

    #[test]
    fn local_path_round_trips() {
        let dir = std::env::temp_dir().join(format!("rmi-img-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tiny.png");
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(apple_png_b64().as_str())
            .unwrap();
        std::fs::write(&path, &bytes).unwrap();

        // bare path
        let image = decode_image_source(path.to_str().unwrap(), false).unwrap();
        assert_eq!(image.bytes, bytes);
        // file:// form
        let uri = format!("file://{}", path.display());
        let image = decode_image_source(&uri, false).unwrap();
        assert_eq!(image.bytes, bytes);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn bad_base64_is_a_clear_error() {
        let error =
            decode_image_source("data:image/png;base64,!!!not-base64!!!", false).unwrap_err();
        assert!(error.contains("base64 decode failed"), "{error}");
    }

    #[test]
    fn empty_source_is_rejected() {
        assert!(decode_image_source("   ", false).is_err());
    }

    #[test]
    fn non_image_base64_fails_at_decode_not_at_parse() {
        // Valid base64, not an image: parsing succeeds, image decoding fails.
        // Long enough to clear the base64 pre-check's length floor (32), so
        // this exercises the base64 branch rather than the file branch.
        let bytes = base64::engine::general_purpose::STANDARD
            .encode(b"this is definitely not an image, only plain text");
        let image = decode_image_source(&bytes, false).unwrap();
        assert!(decode_image_bytes(&image.bytes).is_err(), "not an image");
    }
}
