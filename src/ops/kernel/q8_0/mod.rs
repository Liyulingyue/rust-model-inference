//! Q8_0 matmul kernel implementation.
//!
//! Phase 2.3 + 2.7-final: Q8_0 is the production hot-path kernel for Qwen3-0.6B.
//! The `forward_prequantized` method accepts a Q8-prequantized input and an
//! `ith`/`nth` row partition so callers can dispatch this inside a
//! `pool.compute` closure for thread-parallel matmul.

use super::Kernel;
pub mod avx2;
pub mod avxvnni;
pub mod dispatch;
pub mod ggml;
pub mod neon;
pub mod parallel;
pub mod scalar;

/// Q8_0 matmul kernel: `output = weight × input`, both as Q8_0 blocks.
///
/// `weight` is laid out as Q8_0 blocks: `[2-byte F16 scale][32 bytes data]`,
/// row-major `[n_out rows × n_in cols]` (n_in must be a multiple of 32).
#[derive(Debug, Clone)]
pub struct Q8Kernel<'a> {
    pub weight: &'a [u8],
}

impl<'a> Q8Kernel<'a> {
    pub fn new(weight: &'a [u8]) -> Self {
        Self { weight }
    }
}

impl Drop for Q8Kernel<'_> {
    fn drop(&mut self) {
        #[cfg(feature = "vulkan")]
        crate::ops::float::release_vulkan_weight(self.weight);
    }
}

impl<'a> Kernel for Q8Kernel<'a> {
    fn weight_bytes(&self) -> Option<&[u8]> {
        Some(self.weight)
    }

    fn forward_prequantized(
        &self,
        input_q8: &[u8],
        input_scales: &[f32],
        output: &mut [f32],
        n_in: usize,
        n_out: usize,
        ith: usize,
        nth: usize,
    ) {
        // Delegate to the production matmul which handles AVX2/NEON/GPU
        // dispatch. This is the kernel the Qwen3 hot path actually wants.
        parallel::matmul_q8_0_quantized_parallel_rows(
            self.weight,
            input_q8,
            input_scales,
            output,
            n_in,
            n_out,
            ith,
            nth,
        );
    }

    fn embedding_lookup(&self, token_id: u32, n_embd: usize, out: &mut [f32]) {
        crate::ops::embedding::embedding_lookup_q8_0(self.weight, token_id, n_embd, out);
    }

    /// The Q8_0 storage can be shared between the activation rows of a
    /// batch: the batched kernel computes the same values as the per-row
    /// one.
    fn batchable_q8_0_bytes(&self) -> Option<&[u8]> {
        Some(self.weight)
    }

    /// ggml-parity path for callers that need bit-exact agreement with
    /// `ggml_vec_dot_q8_0_q8_0`. Uses the same `ith`/`nth` row partition as
    /// the production parallel entry so the threading behaviour is
    /// unchanged; only the per-row arithmetic differs.
    fn forward_prepared_ggml(
        &self,
        _input_f32: &[f32],
        input_q8: &[u8],
        input_scales: &[f32],
        _q8_k: Option<&[crate::ops::quant::BlockQ8K]>,
        output: &mut [f32],
        n_in: usize,
        n_out: usize,
        ith: usize,
        nth: usize,
    ) {
        let (my_start, my_end) = if nth <= 1 || n_out == 0 {
            (0, n_out)
        } else {
            let per_thread = (n_out + nth - 1) / nth;
            let start = ith * per_thread;
            let end = (start + per_thread).min(n_out);
            (start, end)
        };
        if my_start >= my_end {
            return;
        }
        ggml::matmul_q8_0_vs_q8_0_ggml(
            self.weight,
            input_q8,
            input_scales,
            &mut output[my_start..my_end],
            n_in,
            my_start,
            my_end,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The batched kernel must compute exactly what the per-row kernel
    /// computes: only the weight reads are shared.
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn batched_neon_matches_the_per_row_kernel_bitwise() {
        let n_in = 5120;
        let n_out = 96;
        let n_rows = 9;
        let mut state = 0x1234_5678u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let blocks = n_in / 32;
        let weight: Vec<u8> = (0..n_out * blocks * 34)
            .map(|_| (next() & 0xff) as u8)
            .collect();
        let input_q8: Vec<u8> = (0..n_rows * n_in).map(|_| (next() & 0xff) as u8).collect();
        let input_scales: Vec<f32> = (0..n_rows * blocks)
            .map(|_| (next() & 0xffff) as f32 / 65535.0)
            .collect();

        let mut per_row = vec![0.0f32; n_rows * n_out];
        for row in 0..n_rows {
            dispatch::matmul_q8_0_quantized_range(
                &weight,
                &input_q8[row * n_in..(row + 1) * n_in],
                &input_scales[row * blocks..(row + 1) * blocks],
                &mut per_row[row * n_out..(row + 1) * n_out],
                n_in,
                0,
                n_out,
            );
        }

        let mut batched = vec![0.0f32; n_rows * n_out];
        dispatch::matmul_q8_0_quantized_batched_range(
            &weight,
            &input_q8,
            &input_scales,
            &mut batched,
            n_in,
            n_out,
            n_rows,
            0,
            1,
        );

        for (index, (a, b)) in per_row.iter().zip(batched.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "row {index} differs: per-row {a} vs batched {b}"
            );
        }
    }

    /// The prepared-rows path must produce what the per-row path produces.
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn prepared_rows_batched_matches_the_per_row_path() {
        // Two rows is the smallest batch the batched path takes, and the
        // row counts around the tile size cover both the full tiles and the
        // short tail tile.
        for n_rows in [2, 3, 5, 8, 9] {
            prepared_rows_batched_matches_the_per_row_path_for(n_rows);
        }
    }

    fn prepared_rows_batched_matches_the_per_row_path_for(n_rows: usize) {
        let n_in = 5120;
        let n_out = 64;
        let mut state = 0xfeed_beefu64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let blocks = n_in / 32;
        let mut weight_bytes = Vec::with_capacity(n_out * blocks * 34);
        for _ in 0..n_out * blocks {
            weight_bytes.extend_from_slice(&crate::ops::f32_to_f16(0.01).to_le_bytes());
            for _ in 0..32 {
                weight_bytes.push((next() & 0xff) as u8);
            }
        }
        let input: Vec<f32> = (0..n_rows * n_in)
            .map(|_| (next() & 0xffff) as f32 / 65535.0 - 0.5)
            .collect();
        let weight =
            crate::ops::kernel::Weight::from_quantized(crate::ops::kernel::QuantizedTensor::Q8_0 {
                data: &weight_bytes,
                n_cols: n_in,
                n_rows: n_out,
            });

        let pool = crate::core::thread_pool::ComputePool::new(2);
        let mut prepared = crate::ops::kernel::PreparedRows::new(n_rows, n_in);
        prepared.prepare(&input, n_rows, n_in, true, false).unwrap();
        let mut batched = vec![0.0f32; n_rows * n_out];
        prepared
            .matmul_group(&input, [(&weight, &mut batched[..])], &pool)
            .unwrap();

        // The reference: the per-row kernel over the same activation.
        let mut per_row = vec![0.0f32; n_rows * n_out];
        for row in 0..n_rows {
            let q8 = &prepared.q8[row * n_in..(row + 1) * n_in];
            let scales = &prepared.scales[row * blocks..(row + 1) * blocks];
            dispatch::matmul_q8_0_quantized_range(
                &weight_bytes,
                q8,
                scales,
                &mut per_row[row * n_out..(row + 1) * n_out],
                n_in,
                0,
                n_out,
            );
        }

        for (index, (a, b)) in per_row.iter().zip(batched.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "element {index} differs: per-row {a} vs prepared {b}"
            );
        }
    }

    /// Build a Q8_0 block for a single 32-element row with all values = `value`.
    /// Returns 34 bytes: 2-byte F16 scale + 32-byte data (all same int8 value).
    fn q8_0_uniform_row(value: i8, scale: f32) -> Vec<u8> {
        let mut row = Vec::with_capacity(34);
        let scale_bits = crate::ops::f32_to_f16(scale).to_le_bytes();
        row.extend_from_slice(&scale_bits);
        for _ in 0..32 {
            row.push(value as u8);
        }
        row
    }

    #[test]
    fn q8_kernel_uniform_weights_sum_to_dot_product() {
        let mut weight = Vec::new();
        weight.extend(q8_0_uniform_row(1, 1.0));

        let input_q8 = vec![1u8; 32];
        let input_scales = vec![1.0f32];

        let mut output = [0.0f32; 1];
        let kernel = Q8Kernel::new(&weight);
        kernel.forward_prequantized(&input_q8, &input_scales, &mut output, 32, 1, 0, 1);

        assert_eq!(output, [32.0]);
    }

    #[test]
    fn q8_kernel_multi_row() {
        let mut weight = Vec::new();
        weight.extend(q8_0_uniform_row(1, 1.0));
        weight.extend(q8_0_uniform_row(1, 1.0));

        let input_q8 = vec![1u8; 32];
        let input_scales = vec![1.0f32];

        let mut output = [0.0f32; 2];
        let kernel = Q8Kernel::new(&weight);
        kernel.forward_prequantized(&input_q8, &input_scales, &mut output, 32, 2, 0, 1);

        assert_eq!(output, [32.0, 32.0]);
    }

    #[test]
    fn q8_kernel_scale_applied() {
        let mut weight = Vec::new();
        weight.extend(q8_0_uniform_row(1, 0.5));

        let input_q8 = vec![1u8; 32];
        let input_scales = vec![1.0f32];

        let mut output = [0.0f32; 1];
        let kernel = Q8Kernel::new(&weight);
        kernel.forward_prequantized(&input_q8, &input_scales, &mut output, 32, 1, 0, 1);

        assert_eq!(output, [16.0]);
    }

    #[test]
    fn q8_kernel_forward_quantizes_internally() {
        let weight = q8_0_uniform_row(1, 1.0);
        let input = vec![1.0f32; 32];
        let mut output = [0.0f32; 1];

        let kernel = Q8Kernel::new(&weight);
        kernel.forward(&input, &mut output, 32, 1);

        assert!((output[0] - 32.0).abs() < 5e-3, "got {}", output[0]);
    }

    #[test]
    fn q8_kernel_thread_partition() {
        // Per-worker row ownership is the CPU contract. GPU dispatch owns the whole output.
        let _cpu = crate::compute::ComputePolicy::Cpu.enter_legacy_scope();
        // 4 rows; simulate 2 threads each writing to its own partition
        // of the same full output slice.
        let mut weight = Vec::new();
        for _ in 0..4 {
            weight.extend(q8_0_uniform_row(1, 1.0));
        }

        let input_q8 = vec![1u8; 32];
        let input_scales = vec![1.0f32];
        let kernel = Q8Kernel::new(&weight);

        // Full 4-element output; threads 0 and 1 write to disjoint halves.
        let mut out_a = [0.0f32; 4];
        kernel.forward_prequantized(&input_q8, &input_scales, &mut out_a, 32, 4, 0, 2);
        // Reset and have thread 1 only (ith=1, nth=2) write its half.
        let mut out_b = [0.0f32; 4];
        kernel.forward_prequantized(&input_q8, &input_scales, &mut out_b, 32, 4, 1, 2);

        assert_eq!(out_a, [32.0, 32.0, 0.0, 0.0]);
        // Thread 1 writes to indices 2..4.
        assert_eq!(out_b[0..2], [0.0, 0.0]);
        assert_eq!(out_b[2..4], [32.0, 32.0]);
    }
}
