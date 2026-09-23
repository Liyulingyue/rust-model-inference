//! Q1_0 block matmul kernel implementation.
//!
//! Q1_0 uses 128-element blocks with 18-byte layout
//! (2-byte F16 scale + 16-byte bitfield, 1 bit per element).
//! Each element dequantizes to `bit ? d : -d`.

use super::Kernel;
#[cfg(target_arch = "x86_64")]
pub mod avx2;
#[cfg(target_arch = "aarch64")]
pub mod neon;
pub mod scalar;

pub use scalar::matmul_q1_0_scalar_range;

#[derive(Debug, Clone, Copy)]
pub struct Q1_0Kernel<'a> {
    pub weight: &'a [u8],
}

impl<'a> Q1_0Kernel<'a> {
    pub const BLOCK_ELEMENTS: usize = 128;
    pub const BLOCK_BYTES: usize = 18;

    pub fn new(data: &'a [u8], _n_in: usize, _n_out: usize) -> Self {
        Self { weight: data }
    }
}

impl<'a> Kernel for Q1_0Kernel<'a> {
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
        #[cfg(target_arch = "x86_64")]
        {
            if crate::ops::has_avx2_fma() {
                let per_thread = (n_out + nth - 1) / nth;
                let my_start = ith * per_thread;
                let my_end = (my_start + per_thread).min(n_out);
                if my_start >= my_end {
                    return;
                }
                unsafe {
                    avx2::matmul_q1_0_vs_q8_0_avx2(
                        self.weight,
                        input_q8,
                        input_scales,
                        &mut output[my_start..my_end],
                        n_in,
                        my_start,
                        my_end,
                    );
                    return;
                }
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                let per_thread = (n_out + nth - 1) / nth;
                let my_start = ith * per_thread;
                let my_end = (my_start + per_thread).min(n_out);
                if my_start < my_end {
                    unsafe {
                        neon::matmul_q1_0_vs_q8_0_neon(
                            self.weight,
                            input_q8,
                            input_scales,
                            &mut output[my_start..my_end],
                            n_in,
                            my_start,
                            my_end,
                        );
                    }
                    return;
                }
            }
        }
        matmul_q1_0_scalar_range(
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
        crate::ops::embedding::embedding_lookup_q1_0(self.weight, token_id, n_embd, out);
    }
}
