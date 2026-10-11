// Hadamard-style MLP (Cactus-Compute sketch).
//
// Input hidden 512 splits into 32 chunks of 16, then re-chunks into
// 16 chunks of 32, alternating with per-channel diagonal conditioner
// `cond = x @ cond_u^T @ cond_v^T` (shape [B, T, HIDDEN]).
//
// Shapes per layer:
//   w1a, w2a, w3a: [NUM_LAYERS, 16, 16]
//   w1b, w2b, w3b: [NUM_LAYERS, 32, 32]
//   d1, d2, d3, d4: [NUM_LAYERS, HIDDEN]
//   b2:            [NUM_LAYERS, HIDDEN]
//   cond_u:        [NUM_LAYERS, 8, HIDDEN]
//   cond_v:        [NUM_LAYERS, HIDDEN, 8]

use super::HIDDEN;

#[derive(Debug)]
pub struct HadamardLayer {
    pub w1a: Vec<f32>,
    pub w1b: Vec<f32>,
    pub w2a: Vec<f32>,
    pub w2b: Vec<f32>,
    pub w3a: Vec<f32>,
    pub w3b: Vec<f32>,
    pub d1: Vec<f32>,
    pub d2: Vec<f32>,
    pub d3: Vec<f32>,
    pub d4: Vec<f32>,
    pub b2: Vec<f32>,
    pub cond_u: Vec<f32>,
    pub cond_v: Vec<f32>,
}

const CHUNK_16: usize = 16;
const CHUNK_32: usize = 32;
const CHUNKS_16: usize = HIDDEN / CHUNK_16; // 32
const CHUNKS_32: usize = HIDDEN / CHUNK_32; // 16
const COND_RANK: usize = 8;

/// `input` is x of shape [HIDDEN]. `layer.cond_u` and `layer.cond_v`
/// are already the **layer-row** slices (8 × HIDDEN and HIDDEN × 8),
/// so no per-layer offset is applied here.
#[inline]
fn conditioner(input: &[f32], layer: &HadamardLayer, out: &mut [f32]) {
    // step 1: x @ cond_u^T → [COND_RANK]
    let mut mid = [0.0f32; COND_RANK];
    for r in 0..COND_RANK {
        let mut acc = 0.0f32;
        for i in 0..HIDDEN {
            acc += input[i] * layer.cond_u[r * HIDDEN + i];
        }
        mid[r] = acc;
    }
    // step 2: mid @ cond_v^T → [HIDDEN]
    for o in 0..HIDDEN {
        let mut acc = 0.0f32;
        for r in 0..COND_RANK {
            acc += mid[r] * layer.cond_v[o * COND_RANK + r];
        }
        out[o] = acc;
    }
}

/// Apply a single [16, 16] matrix to every 16-chunk of `input`.
/// The weight is shared across chunks (per-layer Hadamard matrix).
/// Inner product matches `chunk @ w.T` (PyTorch-style), i.e. `w[o, k]`.
#[inline]
fn matmul_chunked_16(weight: &[f32], input: &[f32], out: &mut [f32]) {
    debug_assert_eq!(weight.len(), CHUNK_16 * CHUNK_16);
    for c in 0..CHUNKS_16 {
        let base = c * CHUNK_16;
        for o in 0..CHUNK_16 {
            let mut acc = 0.0f32;
            for k in 0..CHUNK_16 {
                acc += input[base + k] * weight[o * CHUNK_16 + k];
            }
            out[base + o] = acc;
        }
    }
}

/// Apply a single [32, 32] matrix to every 32-chunk of `input`.
#[inline]
fn matmul_chunked_32(weight: &[f32], input: &[f32], out: &mut [f32]) {
    debug_assert_eq!(weight.len(), CHUNK_32 * CHUNK_32);
    for c in 0..CHUNKS_32 {
        let base = c * CHUNK_32;
        for o in 0..CHUNK_32 {
            let mut acc = 0.0f32;
            for k in 0..CHUNK_32 {
                acc += input[base + k] * weight[o * CHUNK_32 + k];
            }
            out[base + o] = acc;
        }
    }
}

pub fn hadamard_mlp(
    input: &[f32],
    layer: &HadamardLayer,
    _layer_idx: usize,
    batch: usize,
    seq_len: usize,
    output: &mut [f32],
) {
    debug_assert_eq!(input.len(), batch * seq_len * HIDDEN);
    debug_assert_eq!(output.len(), batch * seq_len * HIDDEN);

    // layer d1..d4/b2 are already per-layer rows (load_layer_kernel_nd)

    for b in 0..batch {
        for t in 0..seq_len {
            let base = (b * seq_len + t) * HIDDEN;
            let x = &input[base..base + HIDDEN];

            let mut cond = [0.0f32; HIDDEN];
            conditioner(x, layer, &mut cond);

            let mut s1 = [0.0f32; HIDDEN];
            matmul_chunked_16(&layer.w1a, x, &mut s1);
            for i in 0..HIDDEN {
                s1[i] += layer.d1[i] * cond[i];
            }

            let mut s2 = [0.0f32; HIDDEN];
            matmul_chunked_32(&layer.w1b, &s1, &mut s2);
            for i in 0..HIDDEN {
                s2[i] += layer.d2[i] * cond[i];
            }

            let mut s3 = [0.0f32; HIDDEN];
            matmul_chunked_16(&layer.w2a, &s2, &mut s3);
            for i in 0..HIDDEN {
                s3[i] += layer.d3[i] * cond[i];
            }

            let mut s4 = [0.0f32; HIDDEN];
            matmul_chunked_32(&layer.w2b, &s3, &mut s4);
            for i in 0..HIDDEN {
                s4[i] += layer.d4[i] * cond[i];
            }

            let mut s5 = [0.0f32; HIDDEN];
            matmul_chunked_16(&layer.w3a, &s4, &mut s5);
            let mut s6 = [0.0f32; HIDDEN];
            matmul_chunked_32(&layer.w3b, &s5, &mut s6);
            for i in 0..HIDDEN {
                output[base + i] = s6[i] + layer.b2[i];
            }
        }
    }
}
