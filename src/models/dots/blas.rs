//! CBLAS SGEMM dispatch for `dots.tts`.
//!
//! On macOS, links against Apple's Accelerate framework; on Linux x86_64
//! with the `openblas` Cargo feature enabled, links against system OpenBLAS.
//! All other targets fall through to the caller-provided scalar fallback
//! (the macro / cfg gate at the call site ensures the symbol is never
//! referenced when no backend is available).
//!
//! The signature matches the CBLAS reference: row-major, single-precision,
//! general matrix multiply with explicit `alpha` / `beta`.
//!
//! Build:
//! - `cargo build`                              -- self-contained, no BLAS.
//! - `cargo build --features openblas`          -- link against libopenblas.so
//!   on Linux x86_64; macOS keeps using Accelerate.
//!
//! Call sites gate the BLAS path with the inline cfg predicate
//!   `#[cfg(any(target_os = "macos", all(feature = "openblas", target_os = "linux", target_arch = "x86_64")))]`
//! (and its negation). There is no shared macro / const: `#[cfg(...)]`
//! attribute position does not accept `macro_rules!` expansion, and a `const`
//! is a runtime value rather than a compile-time predicate.

pub mod sys {
    #[cfg(target_os = "macos")]
    #[link(name = "Accelerate", kind = "framework")]
    unsafe extern "C" {
        pub fn cblas_sgemm(
            order: i32,
            trans_a: i32,
            trans_b: i32,
            m: i32,
            n: i32,
            k: i32,
            alpha: f32,
            a: *const f32,
            lda: i32,
            b: *const f32,
            ldb: i32,
            beta: f32,
            c: *mut f32,
            ldc: i32,
        );
    }

    #[cfg(all(feature = "openblas", target_os = "linux", target_arch = "x86_64"))]
    #[link(name = "openblas")]
    unsafe extern "C" {
        pub fn cblas_sgemm(
            order: i32,
            trans_a: i32,
            trans_b: i32,
            m: i32,
            n: i32,
            k: i32,
            alpha: f32,
            a: *const f32,
            lda: i32,
            b: *const f32,
            ldb: i32,
            beta: f32,
            c: *mut f32,
            ldc: i32,
        );
    }
}
