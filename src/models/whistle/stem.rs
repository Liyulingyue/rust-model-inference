//! Conv stem placeholder.
//!
//! Real Cactus-compute conv stem is layered:
//!
//!   dw_1 (Conv2d 1→128, k=3×3) → pw_1 (Conv2d 128→128, k=1×1)
//!   → dw_2 (Conv2d 1→128, k=3×3) → pw_2 (Conv2d 128→128, k=1×1)
//!   → w (Conv2d 1→128, k=3×3) → out (Linear 1280→512)
//!
//! Followed by encoder and decoder. The placeholder lets us drive
//! encoder/decoder parity tests without first reverse-engineering the
//! spatial conv stack. Real stem is the next phase (Phase 2).

/// `placeholder_stem_output` — runs after the mel pipeline has produced
/// the post-stem hidden. The function is a no-op identity copy.
pub fn placeholder_stem_output(input: &[f32], output: &mut [f32]) {
    debug_assert_eq!(input.len(), output.len());
    output.copy_from_slice(input);
}
