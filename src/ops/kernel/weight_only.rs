use super::Weight;
use crate::core::thread_pool::ComputePool;
use crate::ops::dot_f32;

impl Weight<'_> {
    /// Multiply packed weights by unrounded F32 activations. Decode a weight
    /// row once per batch and keep only one unpacked row per active worker.
    pub(crate) fn matmul_weight_only(&self, input: &[f32], output: &mut [f32], pool: &ComputePool) {
        let (width, columns) = (self.n_in, self.n_out);
        assert!(width > 0 && columns > 0 && columns <= u32::MAX as usize);
        assert!(input.len().is_multiple_of(width) && output.len().is_multiple_of(columns));
        assert_eq!(input.len() / width, output.len() / columns);
        let out = output.as_mut_ptr();
        pool.compute(|worker, workers| {
            let count = columns.div_ceil(workers);
            let start = worker * count;
            let end = columns.min(start + count);
            if start >= end {
                return;
            }
            let mut unpacked = vec![0.0; width];
            for col in start..end {
                self.embedding_lookup(col as u32, &mut unpacked);
                for (row, x) in input.chunks_exact(width).enumerate() {
                    let value = dot_f32(x, &unpacked, width);
                    // Each worker owns disjoint columns in every output row.
                    // The checked shapes bound the offset and compute joins all workers.
                    unsafe { out.add(row * columns + col).write(value) }
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::kernel::QuantizedTensor;
    use crate::GGMLType;

    #[test]
    fn packed_projection_retains_small_activations_across_rows_and_partitions() {
        let (width, columns) = (32, 5);
        let mut bytes = vec![0u8; columns * 34];
        for (col, block) in bytes.chunks_exact_mut(34).enumerate() {
            block[..2].copy_from_slice(&crate::ops::f32_to_f16(0.5).to_le_bytes());
            block[2 + col] = 6;
        }
        let weight = Weight::from_quantized(QuantizedTensor::from_bytes(
            &bytes,
            GGMLType::Q8_0,
            width,
            columns,
        ));
        let mut input = vec![0.0; 3 * width];
        for (row, x) in input.chunks_exact_mut(width).enumerate() {
            x[width - 1] = 20.0; // A Q8 activation block would round the small values to zero.
            for (col, value) in x[..columns].iter_mut().enumerate() {
                *value = (col as f32 - 2.5) * (row + 1) as f32 * 0.001;
            }
        }
        for workers in [1, 3, 8] {
            let mut output = vec![f32::NAN; 3 * columns];
            weight.matmul_weight_only(&input, &mut output, &ComputePool::new(workers));
            for (x, y) in input.chunks_exact(width).zip(output.chunks_exact(columns)) {
                for (actual, input) in y.iter().zip(x) {
                    assert!((actual - 3.0 * input).abs() < 1e-7);
                }
            }
        }
    }
}
