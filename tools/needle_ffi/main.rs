// Standalone FFI tool to the closed Cactus-Compute `libneedle3.so`.
//
// This file is INTENTIONALLY outside the main `rust-model-inference`
// crate. The repo's value proposition is pure-Rust GGUF inference;
// `libneedle3.so` is a closed C++ runtime we only need to:
//
//   1. produce oracle transcripts while the conv stem is being
//      reverse-engineered
//   2. extract encoder hidden states (TODO: --dump-encoder-hidden) as
//      the gold standard for lifting the stem placeholder
//
// It is built with `rustc` directly (no Cargo, no extra deps, just
// stdlib + the system `libc` for `dlopen` / `dlsym`). See
// `tools/needle_ffi/README.md` for the build command.
//
// Usage:
//   needle-ffi <whistle.cact> <audio.wav> [libneedle3.so]
//
// stdout: a single JSON object with `text`, `language`, `ttft_ms`,
// `decode_tps`, `words` — pipe through `jq .text` for the transcript.
//
// Exit codes: 0 success, 2 bad args, 3 load error, 4 transcribe error.

use std::env;
use std::ffi::{c_char, c_float, c_int, CStr, CString};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::OnceLock;

// === FFI types matching the libneedle3.so ABI ============================

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
    #[repr(transparent)]
    struct Handle(*mut std::ffi::c_void);
    impl Handle {
        fn is_null(&self) -> bool {
            self.0.is_null()
        }
    }
    // SAFETY: libneedle3.so is documented as not thread-safe; the dev
    // tool only invokes it from a single thread.
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
                unsafe { cstr_lossy(dlerror()) }
            ));
        }
        Ok(unsafe { std::mem::transmute_copy(&ptr) })
    }

    pub(super) fn open(path: &Path) -> Result<Lib, String> {
        let path_c =
            CString::new(path.to_string_lossy().as_bytes()).map_err(|e| format!("path NUL: {e}"))?;
        let raw = unsafe { dlopen(path_c.as_ptr() as *const c_char, RTLD_LAZY) };
        let handle = Handle(raw);
        if handle.is_null() {
            return Err(format!(
                "dlopen({}): {}",
                path.display(),
                unsafe { cstr_lossy(dlerror()) }
            ));
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
        Err("needle-ffi tool only supports linux x86_64".into())
    }
}

unsafe fn cstr_lossy(ptr: *const c_char) -> String {
    if ptr.is_null() {
        return "<null>".into();
    }
    CStr::from_ptr(ptr).to_string_lossy().into_owned()
}

// === Library handle cached for the lifetime of the process ===============

static LIB: OnceLock<Result<&'static imp::Lib, String>> = OnceLock::new();

fn open_library(path: &Path) -> Result<&'static imp::Lib, String> {
    let cached = LIB.get_or_init(|| {
        imp::open(path)
            .map(|l| Box::leak(Box::new(l)) as &'static imp::Lib)
    });
    cached.as_ref().copied().map_err(|e| e.clone())
}

// I keep `Lib` borrowed by reference everywhere; the Box::leak above
// never runs the destructor (intentional — same lifetime as the process).

// === WAV reader: 16-bit, 16 kHz, mono ====================================

fn read_wav_mono_16k(path: &Path) -> Result<Vec<f32>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    if bytes.len() < 44 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(format!("{}: not a RIFF/WAVE file", path.display()));
    }
    let mut pos = 12;
    let mut sample_rate = 0u32;
    let mut channels = 0u16;
    let mut bits_per_sample = 0u16;
    let mut audio_format = 0u16;
    let mut data_off = 0usize;
    let mut data_len = 0usize;
    while pos + 8 <= bytes.len() {
        let tag = &bytes[pos..pos + 4];
        let size = u32::from_le_bytes([
            bytes[pos + 4],
            bytes[pos + 5],
            bytes[pos + 6],
            bytes[pos + 7],
        ]) as usize;
        if tag == b"fmt " {
            audio_format = u16::from_le_bytes([bytes[pos + 8], bytes[pos + 9]]);
            channels = u16::from_le_bytes([bytes[pos + 10], bytes[pos + 11]]);
            sample_rate = u32::from_le_bytes([
                bytes[pos + 12],
                bytes[pos + 13],
                bytes[pos + 14],
                bytes[pos + 15],
            ]);
            bits_per_sample = u16::from_le_bytes([bytes[pos + 22], bytes[pos + 23]]);
        } else if tag == b"data" {
            data_off = pos + 8;
            data_len = size;
            break;
        }
        pos += 8 + size + (size & 1);
    }
    if data_off == 0 {
        return Err(format!("{}: no `data` chunk", path.display()));
    }
    if sample_rate != 16000 {
        return Err(format!(
            "{}: 16 kHz mono required, got {sample_rate} Hz",
            path.display()
        ));
    }
    if bits_per_sample != 16 {
        return Err(format!(
            "{}: 16-bit PCM required, got {bits_per_sample} bits",
            path.display()
        ));
    }
    if audio_format != 1 {
        return Err(format!(
            "{}: PCM required (format={audio_format})",
            path.display()
        ));
    }
    if channels == 0 || channels > 8 {
        return Err(format!(
            "{}: unsupported channel count {channels}",
            path.display()
        ));
    }
    let samples = data_len / (channels as usize * 2);
    let mut out = Vec::with_capacity(samples);
    for s in 0..samples {
        let mut acc = 0i32;
        for c in 0..channels as usize {
            let p = data_off + (s * channels as usize + c) * 2;
            let raw = i16::from_le_bytes([bytes[p], bytes[p + 1]]);
            acc += raw as i32;
        }
        out.push(acc as f32 / (channels as f32 * 32768.0));
    }
    Ok(out)
}

// === Runtime calls =======================================================

unsafe fn last_error(lib: &imp::Lib) -> String {
    let ptr = (lib.last_error)();
    cstr_lossy(ptr)
}

fn load_cact(lib: &imp::Lib, weights_path: &Path) -> Result<(), String> {
    let data = std::fs::read(weights_path)
        .map_err(|e| format!("read {}: {e}", weights_path.display()))?;
    let rc = unsafe { (lib.load)(data.as_ptr() as *const c_char, data.len() as u64) };
    if rc < 0 {
        return Err(format!(
            "needle_load ({} bytes): {}",
            data.len(),
            unsafe { last_error(lib) }
        ));
    }
    Ok(())
}

/// Returns the raw JSON the runtime wrote into `buf`.
fn transcribe(
    lib: &imp::Lib,
    samples: &[f32],
    language: Option<&str>,
    keywords: Option<&str>,
    word_timestamps: bool,
) -> Result<String, String> {
    let lang = language
        .map(CString::new)
        .transpose()
        .map_err(|e| format!("language: {e}"))?;
    let kw = keywords
        .map(CString::new)
        .transpose()
        .map_err(|e| format!("keywords: {e}"))?;

    let mut buf = vec![0u8; 1 << 20];
    let rc = unsafe {
        (lib.transcribe)(
            samples.as_ptr(),
            samples.len() as c_int,
            lang.as_ref()
                .map_or(std::ptr::null(), |c| c.as_ptr()),
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
    Ok(s[..end].to_string())
}

fn default_lib_path(cact: &Path) -> PathBuf {
    if let Ok(p) = env::var("WHISTLE_NEEDLE_LIB_PATH") {
        return PathBuf::from(p);
    }
    let mut p = cact.to_path_buf();
    p.set_file_name("libneedle3.so");
    p
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 || args.len() > 4 {
        eprintln!("usage: needle-ffi <whistle.cact> <audio.wav> [libneedle3.so]");
        return ExitCode::from(2);
    }
    let cact = PathBuf::from(&args[1]);
    let wav = PathBuf::from(&args[2]);
    let lib_path = args
        .get(3)
        .map(PathBuf::from)
        .unwrap_or_else(|| default_lib_path(&cact));

    let lib = match open_library(&lib_path) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("open libneedle3.so ({lib_path:?}): {e}");
            return ExitCode::from(3);
        }
    };
    if let Err(e) = load_cact(lib, &cact) {
        eprintln!("load {cact:?}: {e}");
        return ExitCode::from(3);
    }
    let samples = match read_wav_mono_16k(&wav) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("read {wav:?}: {e}");
            return ExitCode::from(3);
        }
    };
    eprintln!(
        "needle-ffi: loaded {} samples ({:.2} s) from {}",
        samples.len(),
        samples.len() as f32 / 16000.0,
        wav.display()
    );
    let language = env::var("WHISTLE_LANG").ok();
    let json = match transcribe(lib, &samples, language.as_deref(), None, false) {
        Ok(j) => j,
        Err(e) => {
            eprintln!("transcribe: {e}");
            return ExitCode::from(4);
        }
    };
    println!("{json}");
    ExitCode::SUCCESS
}
