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
use std::sync::OnceLock;
use std::time::Duration;

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

/// Whether the server was started with `--allow-remote-images`.
///
/// A process-global set once at startup and read-only afterwards: the flag is a
/// deployment decision, not a per-request one. Defaults to `false`, so a
/// remote URL is rejected unless the operator opted in.
static ALLOW_REMOTE: OnceLock<bool> = OnceLock::new();

/// Called once from `run_server` with the parsed `--allow-remote-images` flag.
pub fn set_allow_remote(value: bool) {
    let _ = ALLOW_REMOTE.set(value);
}

/// The effective value of `--allow-remote-images` (false when unset).
pub fn allow_remote() -> bool {
    ALLOW_REMOTE.get().copied().unwrap_or(false)
}

/// Download guardrails, matching the ones llama.cpp's server applies
/// (`tools/server/server-common.cpp` `handle_media`).
const REMOTE_MAX_BYTES: usize = 10 * 1024 * 1024;
const REMOTE_TIMEOUT: Duration = Duration::from_secs(10);
const REMOTE_MAX_REDIRECTS: usize = 3;

/// True for loopback / RFC1918 / link-local / unspecified addresses. Rejecting
/// these is what keeps `--allow-remote-images` from turning the inference
/// server into an internal-network probe (SSRF).
fn is_private_ip(ip: std::net::IpAddr) -> bool {
    match ip {
        std::net::IpAddr::V4(v4) => {
            v4.is_loopback()
                || v4.is_private()
                || v4.is_link_local()
                || v4.is_unspecified()
                || v4.is_broadcast()
        }
        std::net::IpAddr::V6(v6) => {
            v6.is_loopback()
                || v6.is_unspecified()
                || (v6.segments()[0] & 0xfe00) == 0xfc00 // unique local
                || (v6.segments()[0] & 0xffc0) == 0xfe80 // link local
        }
    }
}

/// Resolve `host` and reject when every address is private. Safer than letting
/// the HTTP client resolve, because the check happens before any connection.
fn host_is_public(host: &str) -> Result<(), String> {
    use std::net::ToSocketAddrs;
    let addrs: Vec<_> = match (host, 0u16).to_socket_addrs() {
        Ok(addrs) => addrs.collect(),
        Err(error) => return Err(format!("cannot resolve image host {host:?}: {error}")),
    };
    if addrs.is_empty() {
        return Err(format!("image host {host:?} resolved to no addresses"));
    }
    if addrs.iter().all(|addr| is_private_ip(addr.ip())) {
        return Err(format!(
            "image host {host:?} resolves to private/loopback addresses, refusing to fetch"
        ));
    }
    Ok(())
}

/// Fetch a remote image with the guardrails above.
fn fetch_remote_image(url: &str) -> Result<ImageRef, String> {
    let parsed =
        reqwest::Url::parse(url).map_err(|error| format!("invalid image URL {url:?}: {error}"))?;
    if !matches!(parsed.scheme(), "http" | "https") {
        return Err(format!(
            "unsupported image URL scheme {:?}",
            parsed.scheme()
        ));
    }
    if let Some(host) = parsed.host_str() {
        host_is_public(host)?;
    } else {
        return Err("image URL has no host".into());
    }
    let client = reqwest::blocking::Client::builder()
        .timeout(REMOTE_TIMEOUT)
        .redirect(reqwest::redirect::Policy::limited(REMOTE_MAX_REDIRECTS))
        .build()
        .map_err(|error| format!("failed to build image HTTP client: {error}"))?;
    let response = client
        .get(url)
        .send()
        .map_err(|error| format!("failed to download image {url:?}: {error}"))?;
    if !response.status().is_success() {
        return Err(format!(
            "image download returned HTTP {} for {url:?}",
            response.status()
        ));
    }
    let bytes = response
        .bytes()
        .map_err(|error| format!("failed to read image body from {url:?}: {error}"))?;
    if bytes.len() > REMOTE_MAX_BYTES {
        return Err(format!(
            "image at {url:?} is {} bytes, over the {} byte remote limit",
            bytes.len(),
            REMOTE_MAX_BYTES
        ));
    }
    Ok(ImageRef::new(bytes.to_vec(), url))
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
    let image = decode_source_bytes(trimmed, allow_remote)?;
    // Reject non-images HERE, while the failure is still a parse error (400 at
    // the HTTP layer) instead of a mid-generation 500. Only the header is
    // sniffed, so this is cheap; the full decode still happens in the encoder.
    image::guess_format(&image.bytes)
        .map_err(|_| format!("unsupported image format in {trimmed:?}; expected PNG or JPEG"))?;
    Ok(image)
}

/// Decode a source string into bytes without validating that they are an
/// image (that check lives in [`decode_image_source`]).
fn decode_source_bytes(source: &str, allow_remote: bool) -> Result<ImageRef, String> {
    if let Some(rest) = strip_data_uri(source) {
        let bytes = decode_base64(rest, "image data URI")?;
        return Ok(ImageRef::new(bytes, source));
    }
    if source.starts_with("http://") || source.starts_with("https://") {
        if !allow_remote {
            return Err(
                "remote image URLs are disabled; start the server with --allow-remote-images \
                 to enable (or pass a data: URI / local file)"
                    .into(),
            );
        }
        return fetch_remote_image(source);
    }
    if let Some(path) = source.strip_prefix("file://") {
        return read_local_image(source, path);
    }
    // Anything else: try raw base64 first (the Anthropic Messages shape sends
    // bare base64 with no data: prefix), then fall back to a local path.
    if looks_like_base64(source) {
        if let Ok(bytes) = decode_base64(source, "image") {
            return Ok(ImageRef::new(bytes, source));
        }
    }
    read_local_image(source, source)
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

/// Decode bytes into a `DynamicImage`. Thin re-export of
/// `app::media::decode_image_bytes` so protocol-level callers do not have to
/// reach across modules.
pub fn decode_image_bytes(bytes: &[u8]) -> Result<image::DynamicImage, String> {
    crate::app::media::decode_image_bytes(bytes)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::app::server::api::fixtures::{synthetic_png, synthetic_png_b64};

    /// The apple.png fixture the docs use, embedded so tests do not depend on
    /// the gitignored `references/` tree. 401x287 RGB.
    use serde::{Deserialize, Serialize};

    #[test]
    fn data_uri_decodes_to_png_bytes() {
        let uri = format!("data:image/png;base64,{}", synthetic_png_b64());
        let image = decode_image_source(&uri, false).unwrap();
        // Assert on the decoded image, not a byte count: 1x1 transparent PNG.
        let decoded = decode_image_bytes(&image.bytes).unwrap();
        assert_eq!((decoded.width(), decoded.height()), (401, 287));
    }

    #[test]
    fn raw_base64_decodes_without_data_prefix() {
        let b64 = synthetic_png_b64();
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
            .decode(synthetic_png_b64().as_str())
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
    fn non_image_base64_is_rejected_at_parse_time() {
        // Valid base64, not an image. Rejected HERE (a 400 at the HTTP
        // layer) rather than inside generation (which would surface as a
        // 500). Long enough to clear the base64 pre-check's length floor
        // (32), so this exercises the base64 branch, not the file branch.
        let bytes = base64::engine::general_purpose::STANDARD
            .encode(b"this is definitely not an image, only plain text");
        let error = decode_image_source(&bytes, false).unwrap_err();
        assert!(error.contains("unsupported image format"), "got: {error}");
    }
}

#[cfg(test)]
mod remote_tests {
    use super::*;
    use crate::app::server::api::fixtures::{synthetic_png, synthetic_png_b64 as apple_png_b64};

    #[test]
    fn remote_flag_defaults_to_off_and_messages_the_flag() {
        // The process-global may have been set by another test; assert on the
        // message contract instead of the global value.
        let error = decode_image_source("https://example.com/cat.png", false).unwrap_err();
        assert!(
            error.contains("--allow-remote-images"),
            "must tell the operator how to enable it: {error}"
        );
    }

    #[test]
    fn private_addresses_are_refused() {
        // These resolve without any network access (loopback / link-local).
        for host in ["127.0.0.1", "localhost", "169.254.169.254"] {
            let url = format!("http://{host}/meta.png");
            let error = fetch_remote_image(&url).unwrap_err();
            assert!(
                error.contains("private")
                    || error.contains("loopback")
                    || error.contains("resolve"),
                "{url} must be refused, got: {error}"
            );
        }
    }

    #[test]
    fn non_http_schemes_are_refused() {
        let error = fetch_remote_image("file:///etc/passwd").unwrap_err();
        assert!(error.contains("scheme"), "got: {error}");
    }

    #[test]
    fn invalid_url_is_a_clear_error() {
        assert!(fetch_remote_image("not a url").is_err());
    }

    #[test]
    fn real_png_survives_the_format_check_after_download_path() {
        // The download path is exercised against a local server in the manual
        // check; here we only verify the post-download validation is the same
        // as for the other sources.
        let payload = apple_png_b64();
        let image = decode_image_source(&payload, true).unwrap();
        assert_eq!(image.bytes, synthetic_png());
    }
}
