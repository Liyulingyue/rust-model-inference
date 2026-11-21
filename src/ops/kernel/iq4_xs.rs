use super::Kernel;

pub struct IQ4XSKernel<'a> {
    pub weight: &'a [u8],
    pub n_in: usize,
    pub n_out: usize,
}

impl<'a> IQ4XSKernel<'a> {
    pub const BLOCK_ELEMENTS: usize = 256;
    pub const BLOCK_BYTES: usize = 136;

    pub fn new(data: &'a [u8], n_in: usize, n_out: usize) -> Self {
        Self {
            weight: data,
            n_in,
            n_out,
        }
    }
}

pub struct IQ2XXSKernel<'a> {
    pub weight: &'a [u8],
    pub n_in: usize,
    pub n_out: usize,
}

impl<'a> IQ2XXSKernel<'a> {
    pub const BLOCK_ELEMENTS: usize = 256;
    pub const BLOCK_BYTES: usize = 66;
    pub fn new(data: &'a [u8], n_in: usize, n_out: usize) -> Self {
        Self {
            weight: data,
            n_in,
            n_out,
        }
    }
}

pub struct IQ2SKernel<'a> {
    pub weight: &'a [u8],
    pub n_in: usize,
    pub n_out: usize,
}

impl<'a> IQ2SKernel<'a> {
    pub const BLOCK_ELEMENTS: usize = 256;
    pub const BLOCK_BYTES: usize = 82;
    pub fn new(data: &'a [u8], n_in: usize, n_out: usize) -> Self {
        Self {
            weight: data,
            n_in,
            n_out,
        }
    }
}

pub struct IQ3XXSKernel<'a> {
    pub weight: &'a [u8],
    pub n_in: usize,
    pub n_out: usize,
}

impl<'a> IQ3XXSKernel<'a> {
    pub const BLOCK_ELEMENTS: usize = 256;
    pub const BLOCK_BYTES: usize = 98;
    pub fn new(data: &'a [u8], n_in: usize, n_out: usize) -> Self {
        Self {
            weight: data,
            n_in,
            n_out,
        }
    }
}

pub struct IQ1MKernel<'a> {
    pub weight: &'a [u8],
    pub n_in: usize,
    pub n_out: usize,
}

impl<'a> IQ1MKernel<'a> {
    pub const BLOCK_ELEMENTS: usize = 256;
    pub const BLOCK_BYTES: usize = 56;
    pub fn new(data: &'a [u8], n_in: usize, n_out: usize) -> Self {
        Self {
            weight: data,
            n_in,
            n_out,
        }
    }
}

pub struct IQ1SKernel<'a> {
    pub weight: &'a [u8],
    pub n_in: usize,
    pub n_out: usize,
}

impl<'a> IQ1SKernel<'a> {
    pub const BLOCK_ELEMENTS: usize = 256;
    pub const BLOCK_BYTES: usize = 50;
    pub fn new(data: &'a [u8], n_in: usize, n_out: usize) -> Self {
        Self {
            weight: data,
            n_in,
            n_out,
        }
    }
}

impl<'a> Kernel for IQ4XSKernel<'a> {
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
        let per_thread = n_out.div_ceil(nth);
        let start = ith * per_thread;
        let end = (start + per_thread).min(n_out);
        if start >= end {
            return;
        }
        // Interface-completeness path (production stays on `forward_prepared`
        // with Q8_K activations): dequantize the Q8_0 activation and each
        // weight row to f32 and dot them, same shape as the IQ4_NL / Q4_K
        // kernels.
        let input: Vec<f32> = input_q8
            .iter()
            .take(n_in)
            .enumerate()
            .map(|(i, &q)| q as i8 as f32 * input_scales[i / 32])
            .collect();
        let row_bytes = n_in / Self::BLOCK_ELEMENTS * Self::BLOCK_BYTES;
        let mut row = vec![0.0f32; n_in];
        for out_idx in start..end {
            let offset = out_idx * row_bytes;
            crate::ops::quant::dequantize_row_iq4_xs(
                &self.weight[offset..offset + row_bytes],
                &mut row,
            );
            output[out_idx] = row.iter().zip(&input).map(|(x, y)| x * y).sum();
        }
    }

    fn forward_prepared(
        &self,
        input_f32: &[f32],
        _input_q8: &[u8],
        _input_scales: &[f32],
        q8_k: Option<&[crate::ops::quant::BlockQ8K]>,
        output: &mut [f32],
        n_in: usize,
        n_out: usize,
        ith: usize,
        nth: usize,
    ) {
        // GPU path: thread 0 submits one fenced dispatch that covers all
        // rows; the rest of the pool returns. Mirrors the Q4_K kernel's
        // `forward_prepared` GPU hook (`ops::kernel::q4_k.rs`).
        #[cfg(feature = "vulkan")]
        {
            use std::sync::atomic::Ordering;
            static IQ4_XS_GPU_DISABLED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !IQ4_XS_GPU_DISABLED.load(Ordering::Relaxed) && crate::ops::gpu_matmul_active() {
                if let Some(ctx) = crate::ops::get_vulkan_context() {
                    if ith == 0 {
                        match crate::vulkan::matmul_iq4_xs::matmul_iq4_xs(
                            ctx,
                            self.weight,
                            &input_f32[..n_in],
                            1,
                            n_in,
                            n_out,
                            output,
                        ) {
                            Ok(()) => return,
                            Err(crate::vulkan::VulkanError::UnsupportedShape(_)) => {
                                // Shape doesn't fit the cached runtime — fall
                                // back to CPU for this matmul and every
                                // subsequent one (the runtime is shared
                                // process-wide).
                                IQ4_XS_GPU_DISABLED.store(true, Ordering::Relaxed);
                            }
                            Err(_) => {
                                // mark_gpu_broken already happened inside
                                // matmul_iq4_xs; the GPU path is gone.
                                return;
                            }
                        }
                    } else {
                        // GPU covers all rows; this thread has nothing to do.
                        return;
                    }
                }
            }
        }

        let per_thread = n_out.div_ceil(nth);
        let start = ith * per_thread;
        let end = (start + per_thread).min(n_out);
        if start >= end {
            return;
        }
        let row_bytes = n_in / Self::BLOCK_ELEMENTS * Self::BLOCK_BYTES;
        for out_idx in start..end {
            let offset = out_idx * row_bytes;
            if let Some(q8k_buf) = q8_k {
                output[out_idx] = crate::ops::quant::vec_dot_iq4_xs_q8k(
                    &self.weight[offset..offset + row_bytes],
                    q8k_buf,
                );
            } else {
                let owned_q8k = crate::ops::quant::quantize_row_q8_k(&input_f32[..n_in]);
                output[out_idx] = crate::ops::quant::vec_dot_iq4_xs_q8k(
                    &self.weight[offset..offset + row_bytes],
                    &owned_q8k,
                );
            }
        }
    }
}

macro_rules! iq_kernel_impl {
    ($name:ident, $block_bytes:expr, $vec_dot:ident, $dequant_row:ident, $gpu_format:ident) => {
        impl<'a> Kernel for $name<'a> {
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
                let per_thread = n_out.div_ceil(nth);
                let start = ith * per_thread;
                let end = (start + per_thread).min(n_out);
                if start >= end {
                    return;
                }
                // Interface-completeness path (production stays on
                // `forward_prepared` with Q8_K activations): dequantize the
                // Q8_0 activation and each weight row to f32 and dot them,
                // same shape as the IQ4_NL / Q4_K kernels.
                let input: Vec<f32> = input_q8
                    .iter()
                    .take(n_in)
                    .enumerate()
                    .map(|(i, &q)| q as i8 as f32 * input_scales[i / 32])
                    .collect();
                let row_bytes = n_in / 256 * $block_bytes;
                let mut row = vec![0.0f32; n_in];
                for out_idx in start..end {
                    let offset = out_idx * row_bytes;
                    crate::ops::quant::$dequant_row(
                        &self.weight[offset..offset + row_bytes],
                        &mut row,
                    );
                    output[out_idx] = row.iter().zip(&input).map(|(x, y)| x * y).sum();
                }
            }

            fn forward_prepared(
                &self,
                input_f32: &[f32],
                _input_q8: &[u8],
                _input_scales: &[f32],
                q8_k: Option<&[crate::ops::quant::BlockQ8K]>,
                output: &mut [f32],
                n_in: usize,
                n_out: usize,
                ith: usize,
                nth: usize,
            ) {
                let per_thread = n_out.div_ceil(nth);
                let start = ith * per_thread;
                let end = (start + per_thread).min(n_out);
                if start >= end {
                    return;
                }
                let row_bytes = n_in / 256 * $block_bytes;
                for out_idx in start..end {
                    let offset = out_idx * row_bytes;
                    if let Some(q8k_buf) = q8_k {
                        output[out_idx] = crate::ops::quant::$vec_dot(
                            &self.weight[offset..offset + row_bytes],
                            q8k_buf,
                        );
                    } else {
                        let owned_q8k = crate::ops::quant::quantize_row_q8_k(&input_f32[..n_in]);
                        output[out_idx] = crate::ops::quant::$vec_dot(
                            &self.weight[offset..offset + row_bytes],
                            &owned_q8k,
                        );
                    }
                }
            }

            fn embedding_lookup(&self, token_id: u32, n_embd: usize, out: &mut [f32]) {
                let row_bytes = n_embd / 256 * $block_bytes;
                let offset = token_id as usize * row_bytes;
                crate::ops::quant::$dequant_row(&self.weight[offset..offset + row_bytes], out);
            }
        }
    };
}

iq_kernel_impl!(
    IQ2XXSKernel,
    66,
    vec_dot_iq2_xxs_q8k,
    dequantize_row_iq2_xxs,
    IQ2_XXS
);
iq_kernel_impl!(
    IQ2SKernel,
    82,
    vec_dot_iq2_s_q8k,
    dequantize_row_iq2_s,
    IQ2_S
);
iq_kernel_impl!(
    IQ3XXSKernel,
    98,
    vec_dot_iq3_xxs_q8k,
    dequantize_row_iq3_xxs,
    IQ3_XXS
);
iq_kernel_impl!(
    IQ3SKernel,
    110,
    vec_dot_iq3_s_q8k,
    dequantize_row_iq3_s,
    IQ3_S
);
iq_kernel_impl!(
    IQ1MKernel,
    56,
    vec_dot_iq1_m_q8k,
    dequantize_row_iq1_m,
    IQ1_M
);
iq_kernel_impl!(
    IQ1SKernel,
    50,
    vec_dot_iq1_s_q8k,
    dequantize_row_iq1_s,
    IQ1_S
);

pub struct IQ2XSKernel<'a> {
    pub weight: &'a [u8],
    pub n_in: usize,
    pub n_out: usize,
}

impl<'a> IQ2XSKernel<'a> {
    pub const BLOCK_ELEMENTS: usize = 256;
    pub const BLOCK_BYTES: usize = 74;

    pub fn new(data: &'a [u8], n_in: usize, n_out: usize) -> Self {
        Self {
            weight: data,
            n_in,
            n_out,
        }
    }
}

impl<'a> Kernel for IQ2XSKernel<'a> {
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
        let per_thread = n_out.div_ceil(nth);
        let start = ith * per_thread;
        let end = (start + per_thread).min(n_out);
        if start >= end {
            return;
        }
        // Interface-completeness path (production stays on `forward_prepared`
        // with Q8_K activations): dequantize the Q8_0 activation and each
        // weight row to f32 and dot them, same shape as the IQ4_NL / Q4_K
        // kernels.
        let input: Vec<f32> = input_q8
            .iter()
            .take(n_in)
            .enumerate()
            .map(|(i, &q)| q as i8 as f32 * input_scales[i / 32])
            .collect();
        let row_bytes = n_in / Self::BLOCK_ELEMENTS * Self::BLOCK_BYTES;
        let mut row = vec![0.0f32; n_in];
        for out_idx in start..end {
            let offset = out_idx * row_bytes;
            crate::ops::quant::dequantize_row_iq2_xs(
                &self.weight[offset..offset + row_bytes],
                &mut row,
            );
            output[out_idx] = row.iter().zip(&input).map(|(x, y)| x * y).sum();
        }
    }

    fn forward_prepared(
        &self,
        input_f32: &[f32],
        _input_q8: &[u8],
        _input_scales: &[f32],
        q8_k: Option<&[crate::ops::quant::BlockQ8K]>,
        output: &mut [f32],
        n_in: usize,
        n_out: usize,
        ith: usize,
        nth: usize,
    ) {
        #[cfg(feature = "vulkan")]
        {
            use std::sync::atomic::Ordering;
            static GPU_DISABLED: std::sync::atomic::AtomicBool =
                std::sync::atomic::AtomicBool::new(false);
            if !GPU_DISABLED.load(Ordering::Relaxed) && crate::ops::gpu_matmul_active() && ith == 0
            {
                if let Some(ctx) = crate::ops::get_vulkan_context() {
                    match crate::vulkan::matmul_iq::matmul_iq(
                        ctx,
                        crate::vulkan::ops::GpuWeightFormat::IQ2_XS,
                        self.weight,
                        &input_f32[..n_in],
                        1,
                        n_in,
                        n_out,
                        output,
                    ) {
                        Ok(()) => return,
                        Err(crate::vulkan::VulkanError::UnsupportedShape(_)) => {
                            GPU_DISABLED.store(true, Ordering::Relaxed);
                        }
                        Err(_) => return,
                    }
                }
            }
        }
        let per_thread = n_out.div_ceil(nth);
        let start = ith * per_thread;
        let end = (start + per_thread).min(n_out);
        if start >= end {
            return;
        }
        let row_bytes = n_in / Self::BLOCK_ELEMENTS * Self::BLOCK_BYTES;
        for out_idx in start..end {
            let offset = out_idx * row_bytes;
            if let Some(q8k_buf) = q8_k {
                output[out_idx] = crate::ops::quant::vec_dot_iq2_xs_q8k(
                    &self.weight[offset..offset + row_bytes],
                    q8k_buf,
                );
            } else {
                let owned_q8k = crate::ops::quant::quantize_row_q8_k(&input_f32[..n_in]);
                output[out_idx] = crate::ops::quant::vec_dot_iq2_xs_q8k(
                    &self.weight[offset..offset + row_bytes],
                    &owned_q8k,
                );
            }
        }
    }
}

pub struct IQ3SKernel<'a> {
    pub weight: &'a [u8],
    pub n_in: usize,
    pub n_out: usize,
}

impl<'a> IQ3SKernel<'a> {
    pub const BLOCK_ELEMENTS: usize = 256;
    pub const BLOCK_BYTES: usize = 110;

    pub fn new(data: &'a [u8], n_in: usize, n_out: usize) -> Self {
        Self {
            weight: data,
            n_in,
            n_out,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic LCG byte stream (same generator family as
    /// tests/quantized_inference.rs::rand_simple).
    struct RandBytes(u32);

    impl Iterator for RandBytes {
        type Item = u8;
        fn next(&mut self) -> Option<u8> {
            self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
            Some((self.0 >> 24) as u8)
        }
    }

    /// One 256-element IQ super-block with a clean f16 scale followed by
    /// random payload bytes. Grid and sign lookups are masked table reads, so
    /// any payload decodes; the seed is bumped until the dequantized row is
    /// finite and non-zero (random scale fields can assemble an Inf/NaN f16
    /// on IQ1_M).
    fn finite_random_super_block(
        block_bytes: usize,
        seed: u32,
        dequant_row: &dyn Fn(&[u8], &mut [f32]),
    ) -> Vec<u8> {
        for seed in seed..seed + 64 {
            let mut block: Vec<u8> = RandBytes(seed).take(block_bytes).collect();
            block[0..2].copy_from_slice(&crate::ops::f32_to_f16(1.0).to_le_bytes());
            let mut row = vec![0.0f32; 256];
            dequant_row(&block, &mut row);
            if row.iter().all(|v| v.is_finite()) && row.iter().any(|v| *v != 0.0) {
                return block;
            }
        }
        panic!("no finite payload seed found starting at {seed}");
    }

    macro_rules! iq_prequantized_test {
        ($name:ident, $kernel:ident, $dequant:ident, $block_bytes:expr, $seed:expr) => {
            /// `forward_prequantized` must compute the plain f32 dot between
            /// the dequantized weight row and the dequantized Q8_0 activation
            /// — the same contract as the IQ4_NL / Q4_K kernels. With an
            /// all-ones activation the dot is the row sum in identical
            /// order, so the comparison is bit-exact.
            #[test]
            fn $name() {
                let n_in = 256;
                let n_out = 2;
                let dequant =
                    |bytes: &[u8], out: &mut [f32]| crate::ops::quant::$dequant(bytes, out);
                let mut weight = Vec::new();
                for seed in [$seed, $seed.wrapping_add(0x9E37_79B9)] {
                    weight.extend(finite_random_super_block($block_bytes, seed, &dequant));
                }
                let mut expected = vec![0.0f32; n_out];
                for (row_idx, row_bytes) in weight.chunks_exact($block_bytes).enumerate() {
                    let mut row = vec![0.0f32; n_in];
                    dequant(&row_bytes, &mut row);
                    expected[row_idx] = row.iter().sum();
                }

                let kernel = $kernel::new(&weight, n_in, n_out);
                let input_q8 = vec![1u8; n_in];
                let input_scales = vec![1.0f32; n_in / 32];
                let mut output = vec![0.0f32; n_out];
                kernel.forward_prequantized(
                    &input_q8,
                    &input_scales,
                    &mut output,
                    n_in,
                    n_out,
                    0,
                    1,
                );
                for i in 0..n_out {
                    assert_eq!(
                        output[i].to_bits(),
                        expected[i].to_bits(),
                        "row {i}: kernel {} vs dequant+dot {}",
                        output[i],
                        expected[i]
                    );
                }
            }
        };
    }

    iq_prequantized_test!(
        iq2_xxs_prequantized_matches_dequant_dot,
        IQ2XXSKernel,
        dequantize_row_iq2_xxs,
        66,
        0xC0FF_EE01u32
    );
    iq_prequantized_test!(
        iq2_s_prequantized_matches_dequant_dot,
        IQ2SKernel,
        dequantize_row_iq2_s,
        82,
        0xC0FF_EE02u32
    );
    iq_prequantized_test!(
        iq3_xxs_prequantized_matches_dequant_dot,
        IQ3XXSKernel,
        dequantize_row_iq3_xxs,
        98,
        0xC0FF_EE03u32
    );
    iq_prequantized_test!(
        iq3_s_prequantized_matches_dequant_dot,
        IQ3SKernel,
        dequantize_row_iq3_s,
        110,
        0xC0FF_EE04u32
    );
    iq_prequantized_test!(
        iq1_m_prequantized_matches_dequant_dot,
        IQ1MKernel,
        dequantize_row_iq1_m,
        56,
        0xC0FF_EE05u32
    );
    iq_prequantized_test!(
        iq1_s_prequantized_matches_dequant_dot,
        IQ1SKernel,
        dequantize_row_iq1_s,
        50,
        0xC0FF_EE06u32
    );
    iq_prequantized_test!(
        iq2_xs_prequantized_matches_dequant_dot,
        IQ2XSKernel,
        dequantize_row_iq2_xs,
        74,
        0xC0FF_EE07u32
    );
    iq_prequantized_test!(
        iq4_xs_prequantized_matches_dequant_dot,
        IQ4XSKernel,
        dequantize_row_iq4_xs,
        136,
        0xC0FF_EE08u32
    );
}
