//! Fixed-order BF16 CPU projections for the pinned Breeze Oracle.

/// BF16 weights in little-endian order; inputs are BF16 values widened to F32.
/// Keep 32 independent accumulators and the ARM Oracle's reduction tree: a
/// sequential sum or the shared NEON kernel crosses BF16 rounding boundaries.
/// This mirrors Torch 2.9.1 ReducedPrecisionFloatGemvFastPathKernel's no_bfdot
/// path, including its eight-element vector tail and final scalar remainder.
pub(super) fn dot_bf16(weight_bytes: &[u8], input: &[f32]) -> f32 {
    assert_eq!(weight_bytes.len(), input.len() * 2);
    let weight = |index: usize| {
        f32::from_bits(
            u32::from(u16::from_le_bytes([
                weight_bytes[index * 2],
                weight_bytes[index * 2 + 1],
            ])) << 16,
        )
    };
    let mut sums = [0.0f32; 32];
    let vector_end = input.len() / 32 * 32;
    for start in (0..vector_end).step_by(32) {
        for lane in 0..32 {
            sums[lane] = input[start + lane].mul_add(weight(start + lane), sums[lane]);
        }
    }
    for offset in [16, 8, 4] {
        for lane in 0..offset {
            sums[lane] += sums[lane + offset];
        }
    }
    let mut sum = (sums[0] + sums[1]) + (sums[2] + sums[3]);
    let tail_end = input.len() / 8 * 8;
    let mut tail = [0.0f32; 4];
    for start in (vector_end..tail_end).step_by(8) {
        for lane in 0..4 {
            tail[lane] = input[start + lane].mul_add(weight(start + lane), tail[lane]);
            tail[lane] = input[start + lane + 4].mul_add(weight(start + lane + 4), tail[lane]);
        }
    }
    sum += (tail[0] + tail[1]) + (tail[2] + tail[3]);
    for (index, &value) in input.iter().enumerate().skip(tail_end) {
        sum += value * weight(index);
    }
    super::bf(sum)
}

/// Torch's generic ARM BF16 GEMM path uses four independent scalar partial sums.
pub(super) fn dot_bf16_gemm(weight_bytes: &[u8], input: &[f32]) -> f32 {
    assert_eq!(weight_bytes.len(), input.len() * 2);
    let weight = |i: usize| {
        f32::from_bits(
            u32::from(u16::from_le_bytes([
                weight_bytes[i * 2],
                weight_bytes[i * 2 + 1],
            ])) << 16,
        )
    };
    let mut partial = [0.0f32; 4];
    let vector_end = input.len() / 4 * 4;
    for (i, &x) in input.iter().enumerate() {
        partial[if i < vector_end { i & 3 } else { 0 }] += x * weight(i);
    }
    super::bf(((partial[0] + partial[1]) + partial[2]) + partial[3])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_dot_keeps_32_lanes_before_pairwise_reduction() {
        let mut weights = vec![0u8; 64 * 2];
        // Sequential or 16-lane accumulation loses the 1 before cancellation.
        for (index, bits) in [(0, 0x4b80u16), (16, 0x3f80), (32, 0xcb80)] {
            weights[index * 2..index * 2 + 2].copy_from_slice(&bits.to_le_bytes());
        }
        assert_eq!(dot_bf16(&weights, &[1.0; 64]).to_bits(), 1.0f32.to_bits());
        // The remainder also has a vectorized eight-element/four-lane stage.
        let mut weights = vec![0u8; 40 * 2];
        for (index, bits) in [(32, 0x4b80u16), (33, 0x3f80), (36, 0xcb80)] {
            weights[index * 2..index * 2 + 2].copy_from_slice(&bits.to_le_bytes());
        }
        assert_eq!(dot_bf16(&weights, &[1.0; 40]).to_bits(), 1.0f32.to_bits());
        // Exact half-way values must round to the even BF16 mantissa.
        let weights = [0x80, 0x3f, 0x80, 0x3b]; // 1 + 1/256.
        assert_eq!(dot_bf16(&weights, &[1.0; 2]).to_bits(), 1.0f32.to_bits());
    }

    #[test]
    fn transposed_head_uses_four_lanes_and_rounds_its_output_to_bf16() {
        let mut weights = vec![0u8; 64 * 2];
        for (index, bits) in [(0, 0x4b80u16), (16, 0x3f80), (32, 0xcb80)] {
            weights[index * 2..index * 2 + 2].copy_from_slice(&bits.to_le_bytes());
        }
        assert_eq!(dot_bf16_gemm(&weights, &[1.0; 64]).to_bits(), 0);
        assert_eq!(
            dot_bf16_gemm(&[0x80, 0x3f, 0x80, 0x3b], &[1.0; 2]).to_bits(),
            1.0f32.to_bits()
        );
    }
}
