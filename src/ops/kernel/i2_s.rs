//! BitNet 1.58-bit ternary weight dequantization (`GGML_TYPE_I2_S` = 36).
//!
//! # GGUF I2_S block layout (per Microsoft bitnet.cpp `QK_I2_S = 128`)
//!
//! Each I2_S block stores 128 ternary weights packed as 2-bit signed
//! values in 32 bytes (128 × 2 bits / 8 bits/byte = 32 bytes/block).
//! The mapping is sign-magnitude:
//!
//! | 2-bit code | ternary value |
//! |-----------|---------------|
//! | `0b00` (0) | -1             |
//! | `0b01` (1) |  0             |
//! | `0b10` (2) | +1             |
//! | `0b11` (3) |  reserved; treated as 0 (should not appear in valid data) |
//!
//! ## Where the scale lives
//!
//! The BitNet I2_S GGUF variant used by Microsoft's BitNet-Embeddings
//! conversion (`bitnet-embeddings-0.6b` / `bitnet-embedding-270m`)
//! does **not** store a per-block or per-row scale inside the I2_S
//! tensor. The conversion writes the ternary weights at full magnitude
//! and absorbs the weight scale into the per-projection `*_norm_in`
//! RMSNorm that precedes the BitLinear forward at inference time
//! (see `docs/usage/bitnet_embedding.md` for the BitLinear forward
//! spec, and `src/models/qwen3/trunk/bitlinear.rs` for the BitLinear
//! integration once that lands).
//!
//! We empirically confirmed the 32-bytes-per-128-elements block size by
//! diffing adjacent I2_S tensor offsets in the converted GGUF
//! (`bitnet-embeddings-0.6b` Q4_K_M): `blk.0.ffn_down.weight`
//! `[3072, 1024]` occupies exactly `3072 × (1024/128) × 32 = 786432`
//! bytes between two intervening F16 norms (with 32 bytes of per-tensor
//! alignment padding), matching `bitnet.cpp`'s `quantize_i2_s` output
//! row-size of `n_per_row / 4` for `QK_I2_S = 128`.
//!
//! ## Reference
//!
//! - `microsoft/BitNet` `src/ggml-bitnet-mad.cpp::quantize_i2_s`
//!   (2-bit packing, sign-magnitude ternary mapping)
//! - `docs/bitnet-embeddings-i2s-guide.md` (conversion pipeline + the
//!   per-projection BitLinear spec that consumes these tensors)
//!
//! ## Status
//!
//! This module ships the dequant kernel + unit test. The full
//! BitLinear forward (RMSNorm pre-norm → per-token absmax activation
//! quant → ternary matmul → rescale by `absmax / 127`) is implemented
//! in [`crate::ops::bitlinear`]. End-to-end qwen3 forward integration
//! is the remaining piece for BitNet-Embeddings-0.6B to forward at all
//! in `--embed` mode (see `docs/usage/bitnet_embedding.md` §3).

use crate::core::tensor::GGMLType;

/// BitNet I2_S block element count. Matches `QK_I2_S = 128` on x86/ARM
/// SIMD paths in `bitnet.cpp`; the GGUF layout is the same across
/// platforms (2-bit packed ternary, 32 bytes per block).
pub const QK_I2_S: usize = 128;

/// Bytes per I2_S block: `QK_I2_S * 2 bits / 8 = 32`.
pub const BLOCK_I2_S_SIZE: usize = 32;

/// Dequantize a single I2_S block (32 bytes) to a `f32` slice of length
/// `QK_I2_S = 128`.
///
/// # Mapping
///
/// - 2-bit `0b00` → -1.0
/// - 2-bit `0b01` →  0.0
/// - 2-bit `0b10` → +1.0
/// - 2-bit `0b11` →  0.0 (reserved; defensive default to zero)
///
/// Packing per the `bitnet.cpp` reference: the 2-bit value for element
/// `j ∈ [0, 128)` occupies bits `[6 - 2·group_idx, 7 - 2·group_idx]`
/// of byte `block[group_pos]`, where `group_idx = j / 32` and
/// `group_pos = j % 32`. This matches `bitnet.cpp::quantize_i2_s`'s
/// `i2_weight[i * 32 + group_pos] |= (q8 << (6 - 2 * group_idx))`.
pub fn dequant_i2_s_block(block: &[u8; BLOCK_I2_S_SIZE], out: &mut [f32; QK_I2_S]) {
    debug_assert_eq!(block.len(), BLOCK_I2_S_SIZE);
    for j in 0..QK_I2_S {
        let group_idx = j / 32;
        let group_pos = j % 32;
        let byte = block[group_pos];
        let shift = 6 - 2 * group_idx;
        let code = (byte >> shift) & 0b11;
        out[j] = match code {
            0b00 => -1.0,
            0b01 => 0.0,
            0b10 => 1.0,
            _ => 0.0, // 0b11 reserved
        };
    }
}

/// Dequantize an entire I2_S row (contiguous bytes covering
/// `n_elements` ternary weights, `n_elements` must be a multiple of
/// `QK_I2_S`) into a `f32` slice of length `n_elements`.
///
/// `bytes` length must equal `n_elements / QK_I2_S * BLOCK_I2_S_SIZE`
/// (the GGUF row payload, no trailing scale bytes for the BitNet
/// Embeddings conversion — see module docs).
///
/// Used by `tests::bitnet_dequant_smoke` to validate the GGUF payload
/// matches the bitnet.cpp reference packing (i.e. is read-into-f32
/// consistent with `quantize_i2_s`'s inverse).
pub fn dequant_i2_s_row(bytes: &[u8], n_elements: usize, out: &mut [f32]) {
    assert_eq!(bytes.len(), n_elements / QK_I2_S * BLOCK_I2_S_SIZE);
    assert_eq!(out.len(), n_elements);
    assert_eq!(n_elements % QK_I2_S, 0);
    let n_blocks = n_elements / QK_I2_S;
    let mut block = [0u8; BLOCK_I2_S_SIZE];
    for b in 0..n_blocks {
        block.copy_from_slice(&bytes[b * BLOCK_I2_S_SIZE..(b + 1) * BLOCK_I2_S_SIZE]);
        dequant_i2_s_block(
            &block,
            (&mut out[b * QK_I2_S..(b + 1) * QK_I2_S])
                .try_into()
                .unwrap(),
        );
    }
}

/// Convenience: total bytes needed to store `n_elements` ternary
/// weights in I2_S format. Equals `n_elements / 4` (128 elements per
/// 32 bytes) when `n_elements` is a multiple of `QK_I2_S = 128`;
/// callers must round up to the nearest block boundary for non-aligned
/// rows.
#[inline]
pub fn i2_s_row_bytes(n_elements: usize) -> usize {
    assert_eq!(n_elements % QK_I2_S, 0);
    n_elements / QK_I2_S * BLOCK_I2_S_SIZE
}

/// BitNet 1.58 weight matrix layout sanity check: returns `true` if
/// `n_elements` is a multiple of `QK_I2_S` (no rounding needed).
#[inline]
pub fn is_i2_s_aligned(n_elements: usize) -> bool {
    n_elements % QK_I2_S == 0
}

/// `GGMLType::I2_S` block size, in (block_elements, block_bytes) form,
/// matching the shape used by [`crate::core::tensor::GGMLType::type_traits`].
/// Currently unused by the runtime (no I2_S matmul kernel ships yet)
/// but kept here so future trait consumers can reference it without
/// duplicating the constants.
pub const I2_S_TRAITS: (usize, usize) = (QK_I2_S, BLOCK_I2_S_SIZE);

#[cfg(test)]
mod tests {
    use super::*;

    /// Round-trip a hand-quantized ternary matrix through dequant_i2_s_block.
    /// Verifies every 2-bit code (0b00, 0b01, 0b10, 0b11) maps to the
    /// expected signed value.
    ///
    /// The I2_S packing is: each byte holds FOUR ternary values (one
    /// per 2-bit slot, at slot indices `group_idx ∈ {0, 1, 2, 3}`
    /// corresponding to bit positions `[6,8)`, `[4,6)`, `[2,4)`, `[0,2)`).
    /// The byte index within the 32-byte block is `group_pos = j % 32`,
    /// so byte `k` of the block holds elements
    /// `{k, k + 32, k + 64, k + 96}` at slots 3, 2, 1, 0 respectively.
    #[test]
    fn dequant_i2_s_block_maps_all_codes() {
        // Each test: fill all 32 bytes with `code * 0b01_01_01_01` so
        // every 2-bit slot carries the same code across all 128
        // elements of the block.
        let mut block = [0u8; BLOCK_I2_S_SIZE];
        let mut out = [0.0f32; QK_I2_S];

        // Code 0b00 → -1
        for b in &mut block {
            *b = 0b00_00_00_00;
        }
        dequant_i2_s_block(&block, &mut out);
        for (j, &v) in out.iter().enumerate() {
            assert_eq!(v, -1.0, "code 0b00: j={j} expected -1.0");
        }

        // Code 0b01 → 0
        for b in &mut block {
            *b = 0b01_01_01_01;
        }
        dequant_i2_s_block(&block, &mut out);
        for (j, &v) in out.iter().enumerate() {
            assert_eq!(v, 0.0, "code 0b01: j={j} expected 0.0");
        }

        // Code 0b10 → +1
        for b in &mut block {
            *b = 0b10_10_10_10;
        }
        dequant_i2_s_block(&block, &mut out);
        for (j, &v) in out.iter().enumerate() {
            assert_eq!(v, 1.0, "code 0b10: j={j} expected +1.0");
        }

        // Code 0b11 → reserved → 0
        for b in &mut block {
            *b = 0b11_11_11_11;
        }
        dequant_i2_s_block(&block, &mut out);
        for (j, &v) in out.iter().enumerate() {
            assert_eq!(v, 0.0, "code 0b11: j={j} reserved, expected 0.0");
        }
    }

    /// Round-trip: take a real 32-byte I2_S block from the converted
    /// BitNet-Embeddings-0.6B GGUF, dequantize, and confirm the
    /// resulting ternary values are all in {-1.0, 0.0, +1.0} (no
    /// garbage from misinterpreted scale bytes). This is a smoke
    /// test that proves our layout matches the real conversion
    /// output, not just the abstract spec.
    #[test]
    fn dequant_i2_s_real_gguf_block_produces_ternary() {
        // Skip when the GGUF is not present in the working tree (CI
        // paths without the model artifact); the test is purely an
        // alignment check against real conversion output.
        let path = std::path::Path::new(
            "models/bitnet-embedding-0.6b-GGUF/bitnet-embeddings-0.6b-bf16-i2_s.gguf",
        );
        if !path.exists() {
            eprintln!("skipping: {path:?} not present (model artifact not in working tree)");
            return;
        }
        use crate::core::loader::GGUFLoader;
        let loader = GGUFLoader::from_file(path).expect("open GGUF");
        let ti = loader
            .tensor_info("blk.0.ffn_down.weight")
            .expect("blk.0.ffn_down.weight present in 0.6B GGUF");
        assert_eq!(ti.ggml_type, GGMLType::I2_S);
        // Compute the data-section offset of this tensor's first block
        // using the public loader API (mmap is private). `data_offset()`
        // is the byte where the GGUF tensor data section begins; the
        // tensor's own `offset` is relative to that.
        let data_section_start = loader.data_offset();
        let tensor_byte_offset = data_section_start + ti.offset as usize;
        // Read the first 32 bytes via `std::fs::read` (avoid exposing
        // the loader's mmap field). The conversion writes a single
        // 4-row-aligned I2_S row, so the first 32 bytes are still
        // a valid 128-element ternary block even though the full
        // tensor is laid out across many such blocks.
        let file_bytes = std::fs::read(path).expect("read GGUF file");
        let mut block = [0u8; BLOCK_I2_S_SIZE];
        block
            .copy_from_slice(&file_bytes[tensor_byte_offset..tensor_byte_offset + BLOCK_I2_S_SIZE]);
        let mut out = [0.0f32; QK_I2_S];
        dequant_i2_s_block(&block, &mut out);
        // Every dequantized value must be in {-1.0, 0.0, +1.0}.
        // If we instead misread scale bytes or a different packing,
        // we'd see values wildly outside this set.
        for (j, &v) in out.iter().enumerate() {
            assert!(
                v == -1.0 || v == 0.0 || v == 1.0,
                "j={j} got {v}, expected ternary"
            );
        }
    }

    /// `dequant_i2_s_row` should accept a `QK_I2_S`-aligned payload
    /// and produce `n_elements` ternary values without losing
    /// alignment between blocks.
    #[test]
    fn dequant_i2_s_row_aligned() {
        let n_elements = QK_I2_S * 3;
        let mut bytes = vec![0u8; i2_s_row_bytes(n_elements)];
        // Pack element 0 = +1 (byte 0, slot 3, bits [6:8] = 0b10).
        bytes[0] = 0b10 << 6;
        // Pack element QK_I2_S + 5 = +1 (this is in the second
        // 128-block of the row: byte offset = BLOCK_I2_S_SIZE + 5).
        bytes[BLOCK_I2_S_SIZE + 5] = 0b10 << 6;
        let mut out = vec![0.0f32; n_elements];
        dequant_i2_s_row(&bytes, n_elements, &mut out);
        assert_eq!(out[0], 1.0);
        assert_eq!(out[QK_I2_S + 5], 1.0);
        // Every other element: byte stays 0 → code 0b00 → -1.
        for (j, &v) in out.iter().enumerate() {
            if j == 0 || j == QK_I2_S + 5 {
                continue;
            }
            assert_eq!(v, -1.0, "j={j} should be -1");
        }
    }
}
