//! Model-aware loading for the NariQ v1 block-Hadamard GGUF releases.
//! A coded weight is a matrix, its two transforms and optional permutations.
mod decode;
mod load;

use super::Kernel;
pub(crate) use load::{is_coded, load_weight};

pub struct TrellisWeight<'a> {
    pub(crate) bands: Vec<Band<'a>>,
    // Original token -> row in the concatenation of frequency bands.
    vocab_gather: Option<Vec<u32>>,
}

pub(crate) struct Band<'a> {
    pub(crate) codes: &'a [u8],
    pub(crate) storage_type: crate::core::tensor::GGMLType,
    pub(crate) rows: usize,
    pub(crate) in_block: usize,
    pub(crate) out_block: usize,
    pub(crate) s16: Vec<f32>,
    pub(crate) pre: Vec<f32>,
    pub(crate) post: Vec<f32>,
    pub(crate) pin: Option<Vec<u32>>,
    // Stored row -> final output index (composes pout and the vocabulary permutation).
    pub(crate) scatter: Vec<u32>,
}

fn hadamard(values: &mut [f32], block: usize) {
    if block == 1 {
        return;
    }
    let scale = (block as f32).sqrt().recip();
    for chunk in values.chunks_exact_mut(block) {
        let mut stride = 1;
        while stride < block {
            for pair in chunk.chunks_exact_mut(2 * stride) {
                let (low, high) = pair.split_at_mut(stride);
                for (a, b) in low.iter_mut().zip(high) {
                    (*a, *b) = (*a + *b, *a - *b);
                }
            }
            stride *= 2;
        }
        chunk.iter_mut().for_each(|v| *v *= scale);
    }
}

impl Band<'_> {
    pub(crate) fn bits(&self) -> usize {
        self.storage_type
            .trellis_bits()
            .expect("load_band requires a TRL type")
    }

    fn tile_row(&self, row: usize) -> &[u8] {
        let stride = self.pre.len() * 2 * self.bits();
        &self.codes[(row / 16) * stride..(row / 16 + 1) * stride]
    }

    fn project(&self, input: &[f32], output: &mut [f32], first: usize) {
        // A Hadamard couples physical rows, but workers own logical rows after
        // the vocabulary and output permutations. A boundary block can thus
        // be evaluated by several workers, each writing only its own slice.
        let group = self.out_block.max(16);
        let owned = first..first + output.len();
        let mut x: Vec<_> = self
            .pre
            .iter()
            .enumerate()
            .map(|(j, &s)| s * input[self.pin.as_ref().map_or(j, |p| p[j] as usize)])
            .collect();
        hadamard(&mut x, self.in_block);
        for (block, &scale) in x.chunks_exact_mut(16).zip(&self.s16) {
            block.iter_mut().for_each(|v| *v *= scale);
        }
        let mut y = vec![0.0; group];
        for (g, destinations) in self.scatter.chunks_exact(group).enumerate() {
            if !destinations
                .iter()
                .any(|&row| owned.contains(&(row as usize)))
            {
                continue;
            }
            let base = g * group;
            for (t, block) in y.chunks_exact_mut(16).enumerate() {
                decode::dot16(self.tile_row(base + 16 * t), self.bits(), &x, block);
            }
            hadamard(&mut y, self.out_block);
            for (i, (&v, &dst)) in y.iter().zip(destinations).enumerate() {
                let dst = dst as usize;
                if owned.contains(&dst) {
                    output[dst - first] = v * self.post[base + i];
                }
            }
        }
    }
}

impl Kernel for TrellisWeight<'_> {
    #[cfg(feature = "vulkan")]
    fn trellis(&self) -> Option<&TrellisWeight<'_>> {
        Some(self)
    }

    fn forward_prepared(
        &self,
        input: &[f32],
        _: &[u8],
        _: &[f32],
        _: Option<&[crate::ops::quant::BlockQ8K]>,
        output: &mut [f32],
        _: usize,
        n_out: usize,
        ith: usize,
        nth: usize,
    ) {
        let per_worker = n_out.div_ceil(nth);
        let first = (ith * per_worker).min(n_out);
        let last = (first + per_worker).min(n_out);
        if first == last {
            return;
        }
        for band in &self.bands {
            band.project(input, &mut output[first..last], first);
        }
    }

    fn forward_prequantized(
        &self,
        input: &[u8],
        scales: &[f32],
        output: &mut [f32],
        n_in: usize,
        n_out: usize,
        ith: usize,
        nth: usize,
    ) {
        let x: Vec<_> = input[..n_in]
            .iter()
            .enumerate()
            .map(|(i, &v)| (v as i8) as f32 * scales[i / 32])
            .collect();
        self.forward_prepared(&x, &[], &[], None, output, n_in, n_out, ith, nth);
    }

    fn embedding_lookup(&self, token: u32, _: usize, out: &mut [f32]) {
        let mut row = self
            .vocab_gather
            .as_ref()
            .map_or(token, |p| p[token as usize]) as usize;
        for band in &self.bands {
            if row < band.rows {
                decode::embedding_row(band.tile_row(row), band.bits(), row % 16, &band.s16, out);
                hadamard(out, band.in_block);
                for (v, &pre) in out.iter_mut().zip(&band.pre) {
                    *v *= pre * band.post[row];
                }
                return;
            }
            row -= band.rows;
        }
        unreachable!("token outside the validated vocabulary")
    }
}

#[cfg(test)]
mod tests;
