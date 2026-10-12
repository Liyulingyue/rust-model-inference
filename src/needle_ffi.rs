//! FFI bindings to the closed Cactus-compute `libneedle3.so` runtime
//! loaded via raw `dlopen`/`dlsym` (no `libloading` dep).
//!
//! The Cactus runtime ships only as a closed binary (loaded via
//! `dlopen`) — we expose its symbols here so the Rust CLI can dispatch
//! to a real Whisper transcription even though our pure-Rust port of
//! the audio encoder + conv stem is still placeholder. The LLM half
//! (encoder + stack + decoder logits) remains a Rust port under
//! `crate::models::whistle`; the FFI path is opt-in via the
//! `--needle-runtime` CLI flag.

use std::ffi::{c_char, c_float, c_int, CStr, CString};
use std::path::Path;
use std::sync::OnceLock;

type NeedleInit = unsafe extern "C" fn(*const c_char, *const c_char, *const c_char) -> c_int;
type NeedleLoad = unsafe extern "C" fn(*const c_char, u64) -> i32;
type NeedleLastError = unsafe extern "C" fn() -> *const c_char;
type NeedleReset = unsafe extern "C" fn();
type NeedleTranscribe = unsafe extern "C" fn(
    *const c_float,
    c_int,
    *const c_char,
    *const c_char,
    c_int,
    *mut c_char,
    c_int,
) -> c_int;

#[cfg(target_os = "linux")]
mod imp {
    use super::*;

    #[derive(Clone, Copy)]
    struct Handle(*mut std::ffi::c_void);
    impl Handle {
        fn is_null(&self) -> bool {
            self.0.is_null()
        }
    }
    // SAFETY: the underlying libneedle3.so runtime is documented as not
    // thread-safe per process; we only call into it from a single worker
    // at a time. The handle is Send so we can stash it inside the
    // `OnceLock<Result<Lib, String>>`.
    unsafe impl Send for Handle {}
    unsafe impl Sync for Handle {}

    extern "C" {
        fn dlopen(filename: *const c_char, flag: c_int) -> *mut std::ffi::c_void;
        fn dlsym(handle: Handle, symbol: *const c_char) -> *mut std::ffi::c_void;
        fn dlclose(handle: Handle) -> c_int;
        fn dlerror() -> *const c_char;
    }
    const RTLD_LAZY: c_int = 0x00001;

    pub(super) struct Lib {
        pub(super) init: NeedleInit,
        pub(super) load: NeedleLoad,
        pub(super) last_error: NeedleLastError,
        pub(super) reset: NeedleReset,
        pub(super) transcribe: NeedleTranscribe,
        _handle: Handle,
    }

    fn lookup<T>(handle: Handle, name: &[u8]) -> Result<T, String> {
        let ptr = unsafe { dlsym(handle, name.as_ptr() as *const c_char) };
        if ptr.is_null() {
            return Err(format!(
                "dlsym {}: {}",
                std::str::from_utf8(name).unwrap_or("?"),
                unsafe {
                    std::str::from_utf8_unchecked(std::ffi::CStr::from_ptr(dlerror()).to_bytes())
                }
            ));
        }
        Ok(unsafe { std::mem::transmute_copy(&ptr) })
    }

    pub(super) fn open(path: &Path) -> Result<Lib, String> {
        let path_c = CString::new(path.to_string_lossy().as_bytes())
            .map_err(|e| format!("path NUL: {e}"))?;
        let raw = unsafe { dlopen(path_c.as_ptr() as *const c_char, RTLD_LAZY) };
        let handle = Handle(raw);
        if handle.is_null() {
            return Err(format!("dlopen({}): {}", path.display(), unsafe {
                std::str::from_utf8_unchecked(std::ffi::CStr::from_ptr(dlerror()).to_bytes())
            }));
        }
        let init: NeedleInit = lookup(handle, b"needle_init\0")?;
        let load: NeedleLoad = lookup(handle, b"needle_load\0")?;
        let last_error: NeedleLastError = lookup(handle, b"needle_last_error\0")?;
        let reset: NeedleReset = lookup(handle, b"needle_reset\0")?;
        let transcribe: NeedleTranscribe = lookup(handle, b"needle_transcribe\0")?;
        Ok(Lib {
            init,
            load,
            last_error,
            reset,
            transcribe,
            _handle: handle,
        })
    }
}

#[cfg(not(target_os = "linux"))]
mod imp {
    use super::*;
    pub(super) struct Lib;
    pub(super) fn open(_path: &Path) -> Result<Lib, String> {
        Err("needle FFI only supported on linux x86_64".into())
    }
}

static LIB: OnceLock<Result<imp::Lib, String>> = OnceLock::new();

fn open_library(path: &Path) -> Result<&'static imp::Lib, String> {
    let cached = LIB.get_or_init(|| imp::open(path));
    cached.as_ref().map_err(|e| e.clone())
}

/// Path to the bundled `libneedle3.so`. Defaults to
/// `models/whistle-probe/libneedle3.so`; override via the
/// `WHISTLE_NEEDLE_LIB_PATH` environment variable.
pub fn lib_path() -> std::path::PathBuf {
    if let Ok(p) = std::env::var("WHISTLE_NEEDLE_LIB_PATH") {
        return std::path::PathBuf::from(p);
    }
    let manifest = std::path::PathBuf::from(
        std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string()),
    );
    manifest
        .join("models")
        .join("whistle-probe")
        .join("libneedle3.so")
}

unsafe fn cstr_to_string(ptr: *const c_char) -> Result<String, String> {
    if ptr.is_null() {
        return Err("null error pointer".into());
    }
    CStr::from_ptr(ptr)
        .to_str()
        .map(|s| s.to_string())
        .map_err(|e| format!("non-utf8 error string: {e}"))
}

unsafe fn last_error(lib: &imp::Lib) -> String {
    let ptr = (lib.last_error)();
    cstr_to_string(ptr).unwrap_or_else(|e| e)
}

/// Load the `.cact` weights.
pub fn load(weights_path: &Path) -> Result<(), String> {
    let path = lib_path();
    let lib = open_library(&path)?;
    let data =
        std::fs::read(weights_path).map_err(|e| format!("read {}: {e}", weights_path.display()))?;
    let rc = unsafe { (lib.load)(data.as_ptr() as *const c_char, data.len() as u64) };
    if rc < 0 {
        return Err(format!("needle_load ({} bytes): {}", data.len(), unsafe {
            last_error(lib)
        }));
    }
    Ok(())
}

/// Run transcription on a 16 kHz mono PCM buffer of `f32` samples in
/// `[-1, 1]`. `language` is one of `en`, `de`, `fr`, `es`, `it`,
/// `nl`, `pl` or `None` to auto-detect. Returns the JSON dict with
/// keys `text`, `language`, `ttft_ms`, `decode_tps`.
pub fn transcribe(
    samples: &[f32],
    language: Option<&str>,
    keywords: Option<&str>,
    word_timestamps: bool,
) -> Result<TranscribeResult, String> {
    let path = lib_path();
    let lib = open_library(&path)?;
    let lang = language
        .map(|s| CString::new(s))
        .transpose()
        .map_err(|e| format!("language: {e}"))?;
    let kw = keywords
        .map(|s| CString::new(s))
        .transpose()
        .map_err(|e| format!("keywords: {e}"))?;

    // Output buffer — 1 MiB is plenty for any single-call JSON.
    let mut buf = vec![0u8; 1 << 20];
    let rc = unsafe {
        (lib.transcribe)(
            samples.as_ptr(),
            samples.len() as c_int,
            lang.as_ref().map_or(std::ptr::null(), |c| c.as_ptr()),
            kw.as_ref().map_or(std::ptr::null(), |c| c.as_ptr()),
            word_timestamps as c_int,
            buf.as_mut_ptr() as *mut c_char,
            buf.len() as c_int,
        )
    };
    if rc < 0 {
        return Err(format!("needle_transcribe: {}", unsafe { last_error(lib) }));
    }
    let s = std::str::from_utf8(&buf).map_err(|e| format!("non-utf8 output: {e}"))?;
    let end = s.find('\0').unwrap_or(s.len());
    let json = &s[..end];
    serde_json::from_str::<TranscribeResult>(json).map_err(|e| format!("json decode {json:?}: {e}"))
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct TranscribeResult {
    pub text: String,
    #[serde(default)]
    pub language: String,
    #[serde(default)]
    pub ttft_ms: f64,
    #[serde(default)]
    pub decode_tps: f64,
    #[serde(default)]
    pub words: Vec<Word>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Word {
    pub word: String,
    pub start: f64,
    pub end: f64,
    pub probability: f64,
}
