//! Lookup tables for IQ2/IQ3/IQ1 quant codebooks and sign masks.
//!
//! All values are transcribed verbatim from `llama.cpp` `ggml-common.h` into
//! plain Rust `const` arrays. No `.h` is parsed at build time and no tables
//! are parsed at runtime; the data lives directly in the binary's read-only
//! data section. If upstream `llama.cpp` ever changes one of these tables,
//! regenerate `iq_tables_data.rs` from the header.

include!("iq_tables_data.rs");

/// Non-linear 4-bit quantization values for IQ4_NL.
/// Mirrors `kvalues_iq4nl` in `llama.cpp`.
pub const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

// Re-export the two constants that have upstream names not matching our
// `IQ<family>_<table>` convention. The other tables already use that name.
pub use {
    KMASK_IQ2XS as IQ2XS_MASK,
    KSIGNS_IQ2XS as IQ2XS_SIGNS,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expected_lengths() {
        assert_eq!(IQ2XS_MASK.len(), 8);
        assert_eq!(IQ2XS_SIGNS.len(), 128);
        assert_eq!(IQ2XS_GRID.len(), 512);
        assert_eq!(IQ2XXS_GRID.len(), 256);
        assert_eq!(IQ2S_GRID.len(), 1024);
        assert_eq!(IQ3S_GRID.len(), 512);
        assert_eq!(IQ3XXS_GRID.len(), 256);
        assert_eq!(IQ1S_GRID.len(), 2048);
        assert_eq!(KVALUES_IQ4NL.len(), 16);
    }
}