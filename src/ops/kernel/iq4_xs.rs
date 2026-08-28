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
        Self { weight: data, n_in, n_out }
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
        Self { weight: data, n_in, n_out }
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
        Self { weight: data, n_in, n_out }
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
        Self { weight: data, n_in, n_out }
    }
}

impl<'a> Kernel for IQ4XSKernel<'a> {
    fn forward_prequantized(
        &self,
        _input_q8: &[u8],
        _input_scales: &[f32],
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
        for o in output.iter_mut() {
            *o = 0.0;
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
        let row_bytes =
            n_in / Self::BLOCK_ELEMENTS * Self::BLOCK_BYTES;
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
    ($name:ident, $block_bytes:expr, $vec_dot:ident, $dequant_row:ident) => {
        impl<'a> Kernel for $name<'a> {
            fn forward_prequantized(
                &self,
                _input_q8: &[u8],
                _input_scales: &[f32],
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
                for o in output.iter_mut() {
                    *o = 0.0;
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
                crate::ops::quant::$dequant_row(
                    &self.weight[offset..offset + row_bytes],
                    out,
                );
            }
        }
    };
}

iq_kernel_impl!(IQ2XXSKernel, 66, vec_dot_iq2_xxs_q8k, dequantize_row_iq2_xxs);
iq_kernel_impl!(IQ2SKernel, 82, vec_dot_iq2_s_q8k, dequantize_row_iq2_s);
iq_kernel_impl!(IQ3XXSKernel, 98, vec_dot_iq3_xxs_q8k, dequantize_row_iq3_xxs);
iq_kernel_impl!(IQ3SKernel, 110, vec_dot_iq3_s_q8k, dequantize_row_iq3_s);

pub struct IQ2XSKernel<'a> {
    pub weight: &'a [u8],
    pub n_in: usize,
    pub n_out: usize,
}

impl<'a> IQ2XSKernel<'a> {
    pub const BLOCK_ELEMENTS: usize = 256;
    pub const BLOCK_BYTES: usize = 74;

    pub fn new(data: &'a [u8], n_in: usize, n_out: usize) -> Self {
        Self { weight: data, n_in, n_out }
    }
}

impl<'a> Kernel for IQ2XSKernel<'a> {
    fn forward_prequantized(
        &self,
        _input_q8: &[u8],
        _input_scales: &[f32],
        output: &mut [f32],
        _n_in: usize,
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
        for o in output.iter_mut() {
            *o = 0.0;
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
        Self { weight: data, n_in, n_out }
    }
}
