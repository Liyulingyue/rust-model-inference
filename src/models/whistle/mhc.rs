// Real multi-lane hyper-connection (`mhc_lanes=4`) from
// cactus-compute/needle/model/architecture.py `Stack._advance`.
//
// The encoder block is wrapped by 4 lanes. Each layer:
//
//   1. broadcasts the input `[B, T, 512]` to `[B, T, 4, 512]`
//   2. rms-units the 4×512 = 2048 dim vector per (B, T)
//   3. computes hpre = sigmoid(a_pre * (nx @ phi_pre) + b_pre + pre_off)
//      where `pre_off = 8 * lane - 4` cycles by lane across layers.
//   4. merges lanes to a single 512 dim `u` via `Σ_lane hpre * lane`
//   5. runs the inner encoder block on `u` (ZCRMSNorm → attn → ZCRMSNorm
//      → attn_gate * skip → ZCRMSNorm → HadamardMLP → skip)
//   6. block output minus input → y
//   7. computes hpost = 2 * sigmoid(a_post * (nx @ phi_post) + b_post + post_off)
//   8. computes the per-(B,T) res mixing matrix via sinkhorn(a_res *
//      (nx @ phi_res).reshape(4, 4) + b_res)
//   9. new_stream = hres @ xf + hpost[..., None] * y[..., None, :]
//  10. takes mean over the 4 lanes → output `[B, T, 512]`.

use crate::ops::norm::rms_norm;

use super::{
    gqa_self_attention, hadamard_mlp, zc_rms_norm_placeholder, HadamardLayer, HEAD_DIM_K,
    HEAD_DIM_Q, HIDDEN, NUM_HEADS, NUM_KV_HEADS, NUM_LAYERS,
};

pub const MHC_LANES: usize = 4;
const PHI_PRE_OUT: usize = MHC_LANES; // 4
const PHI_RES_OUT: usize = MHC_LANES * MHC_LANES; // 16
const PHI_CHANNELS: usize = MHC_LANES * HIDDEN; // 2048

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Sinkhorn-Knopp normalisation on a 4×4 matrix per (B, T). Performs
/// 20 alternating log-row / log-column normalisations in log-space, then
/// a single exp at the end (cactus-compute/needle `_sinkhorn`).
fn sinkhorn(logits: &mut [f32], batch: usize, seq_len: usize) {
    let n = MHC_LANES;
    let iters = 20;
    for bt in 0..(batch * seq_len) {
        let base = bt * (n * n);
        for _ in 0..iters {
            // Row-normalise: subtract per-row logsumexp.
            for r in 0..n {
                let mut max_v = f32::NEG_INFINITY;
                for c in 0..n {
                    let v = logits[base + r * n + c];
                    if v > max_v {
                        max_v = v;
                    }
                }
                let mut sum = 0.0f32;
                for c in 0..n {
                    sum += (logits[base + r * n + c] - max_v).exp();
                }
                let log_z = max_v + sum.ln();
                for c in 0..n {
                    logits[base + r * n + c] -= log_z;
                }
            }
            // Col-normalise: subtract per-col logsumexp.
            for c in 0..n {
                let mut max_v = f32::NEG_INFINITY;
                for r in 0..n {
                    let v = logits[base + r * n + c];
                    if v > max_v {
                        max_v = v;
                    }
                }
                let mut sum = 0.0f32;
                for r in 0..n {
                    sum += (logits[base + r * n + c] - max_v).exp();
                }
                let log_z = max_v + sum.ln();
                for r in 0..n {
                    logits[base + r * n + c] -= log_z;
                }
            }
        }
        // Final exp out of log-space.
        for r in 0..n {
            for c in 0..n {
                logits[base + r * n + c] = logits[base + r * n + c].exp();
            }
        }
    }
}

#[derive(Debug)]
pub struct MhcLayer {
    pub phi_pre: Vec<f32>, // [NUM_LAYERS, PHI_CHANNELS, PHI_PRE_OUT] = [8, 2048, 4]
    pub phi_post: Vec<f32>, // [8, 2048, 4]
    pub phi_res: Vec<f32>, // [8, 2048, 16]
    pub b_pre: Vec<f32>,   // [8, 4]
    pub b_post: Vec<f32>,  // [8, 4]
    pub b_res: Vec<f32>,   // [8, 4, 4]
    pub a_pre: Vec<f32>,   // [8]
    pub a_post: Vec<f32>,  // [8]
    pub a_res: Vec<f32>,   // [8]
}

/// Load mhc weights from the source.
pub fn load_mhc_layer(source: &dyn crate::core::tensor::TensorSource) -> MhcLayer {
    MhcLayer {
        phi_pre: crate::core::tensor::load_f32_tensor(
            source,
            "encoder/mhc_phi_pre",
            &[NUM_LAYERS as u64, PHI_CHANNELS as u64, PHI_PRE_OUT as u64],
        )
        .expect("mhc_phi_pre"),
        phi_post: crate::core::tensor::load_f32_tensor(
            source,
            "encoder/mhc_phi_post",
            &[NUM_LAYERS as u64, PHI_CHANNELS as u64, PHI_PRE_OUT as u64],
        )
        .expect("mhc_phi_post"),
        phi_res: crate::core::tensor::load_f32_tensor(
            source,
            "encoder/mhc_phi_res",
            &[NUM_LAYERS as u64, PHI_CHANNELS as u64, PHI_RES_OUT as u64],
        )
        .expect("mhc_phi_res"),
        b_pre: crate::core::tensor::load_f32_tensor(
            source,
            "encoder/mhc_b_pre",
            &[NUM_LAYERS as u64, PHI_PRE_OUT as u64],
        )
        .expect("mhc_b_pre"),
        b_post: crate::core::tensor::load_f32_tensor(
            source,
            "encoder/mhc_b_post",
            &[NUM_LAYERS as u64, PHI_PRE_OUT as u64],
        )
        .expect("mhc_b_post"),
        b_res: crate::core::tensor::load_f32_tensor(
            source,
            "encoder/mhc_b_res",
            &[NUM_LAYERS as u64, PHI_PRE_OUT as u64, PHI_PRE_OUT as u64],
        )
        .expect("mhc_b_res"),
        a_pre: crate::core::tensor::load_f32_tensor(
            source,
            "encoder/mhc_a_pre",
            &[NUM_LAYERS as u64],
        )
        .expect("mhc_a_pre"),
        a_post: crate::core::tensor::load_f32_tensor(
            source,
            "encoder/mhc_a_post",
            &[NUM_LAYERS as u64],
        )
        .expect("mhc_a_post"),
        a_res: crate::core::tensor::load_f32_tensor(
            source,
            "encoder/mhc_a_res",
            &[NUM_LAYERS as u64],
        )
        .expect("mhc_a_res"),
    }
}

#[inline]
fn lane_offsets(layer_idx: usize, pre_out: &mut [f32; MHC_LANES], post_out: &mut [f32; MHC_LANES]) {
    // lane = (1 at position layer_idx % 4, else 0)
    let active = layer_idx % MHC_LANES;
    for i in 0..MHC_LANES {
        if i == active {
            pre_out[i] = 8.0 - 4.0; // 4.0
            post_out[i] = 0.0;
        } else {
            pre_out[i] = -4.0;
            post_out[i] = -4.0;
        }
    }
}

/// One encoder block on a fresh `[B, T, 512]` input. Broadcasts to
/// `[B, T, 4, 512]`, runs the inner encoder block + mhc gate, and
/// returns `mean(stream, axis=2)` to `[B, T, 512]`.
pub fn forward_one(
    layer: &EncoderLayerWeights<'_>,
    mhc: &MhcLayer,
    x: &[f32],
    layer_idx: usize,
    batch: usize,
    seq_len: usize,
    out: &mut [f32],
) {
    debug_assert_eq!(x.len(), batch * seq_len * HIDDEN);
    debug_assert_eq!(out.len(), batch * seq_len * HIDDEN);

    let plane = batch * seq_len * HIDDEN;
    let mut stream = vec![0.0f32; batch * seq_len * MHC_LANES * HIDDEN];
    // broadcast x → [B, T, 4, 512] (lane-major: x[t, lane, h])
    for bt in 0..(batch * seq_len) {
        let src = &x[bt * HIDDEN..bt * HIDDEN + HIDDEN];
        for lane in 0..MHC_LANES {
            let dst = &mut stream[bt * MHC_LANES * HIDDEN + lane * HIDDEN..][..HIDDEN];
            dst.copy_from_slice(src);
        }
    }
    // Stream is now fully initialised, mut borrow is OK to release.
    let mut new_stream = stream.clone();
    forward_one_stream(
        layer,
        mhc,
        &stream,
        layer_idx,
        batch,
        seq_len,
        &mut new_stream,
    );

    // mean across lanes → [B, T, 512]
    for bt in 0..(batch * seq_len) {
        for h in 0..HIDDEN {
            let mut sum = 0.0f32;
            for lane in 0..MHC_LANES {
                sum += new_stream[bt * MHC_LANES * HIDDEN + lane * HIDDEN + h];
            }
            out[bt * HIDDEN + h] = sum / MHC_LANES as f32;
        }
    }
}

/// One encoder block on a stream `[B, T, 4, 512]` (cumulative). Writes
/// the new stream back into `stream_out` (which must already be sized
/// `batch * seq_len * MHC_LANES * HIDDEN`).
pub fn forward_one_stream(
    layer: &EncoderLayerWeights<'_>,
    mhc: &MhcLayer,
    stream: &[f32],
    layer_idx: usize,
    batch: usize,
    seq_len: usize,
    stream_out: &mut [f32],
) {
    debug_assert_eq!(stream.len(), batch * seq_len * MHC_LANES * HIDDEN);
    debug_assert_eq!(stream_out.len(), batch * seq_len * MHC_LANES * HIDDEN);

    let n = MHC_LANES;
    let c = HIDDEN;

    // Step 1: rms_unit(stream reshape [B, T, 2048])
    // nx shape [B, T, 2048]
    let plane = batch * seq_len;
    let mut nx = vec![0.0f32; plane * n * c];
    for bt in 0..plane {
        let sum_sq = {
            let mut s = 0.0f32;
            for lane in 0..n {
                for h in 0..c {
                    let v = stream[bt * n * c + lane * c + h];
                    s += v * v;
                }
            }
            s
        };
        let inv_rms = 1.0f32 / ((sum_sq / (n * c) as f32) + 1e-6).sqrt();
        for k in 0..(n * c) {
            nx[bt * (n * c) + k] = stream[bt * n * c + k] * inv_rms;
        }
    }

    // Step 2: hpre = sigmoid(a_pre * (nx @ phi_pre) + b_pre + pre_off)
    let phi_pre_off = layer_idx * (n * c) * n;
    let phi_pre_b_off = layer_idx * n;
    let a_pre = mhc.a_pre[layer_idx];
    let pre_off_arr = [0.0f32; MHC_LANES];
    let post_off_arr = [0.0f32; MHC_LANES];
    let mut pre_off_local = pre_off_arr;
    let mut post_off_local = post_off_arr;
    lane_offsets(layer_idx, &mut pre_off_local, &mut post_off_local);

    // hpre shape [B, T, 4]
    let mut hpre = vec![0.0f32; plane * n];
    for bt in 0..plane {
        for lane in 0..n {
            // dot = Σ_k nx[bt, k] * phi_pre[k, lane]
            let mut dot = 0.0f32;
            for k in 0..(n * c) {
                dot += nx[bt * (n * c) + k] * mhc.phi_pre[phi_pre_off + k * n + lane];
            }
            let v = a_pre * dot + mhc.b_pre[phi_pre_b_off + lane] + pre_off_local[lane];
            hpre[bt * n + lane] = sigmoid(v);
        }
    }

    // Step 3: u = sum(hpre[..., None] * xf, axis=2)  — [B, T, 512]
    let mut u = vec![0.0f32; plane * c];
    for bt in 0..plane {
        for h in 0..c {
            let mut acc = 0.0f32;
            for lane in 0..n {
                acc += hpre[bt * n + lane] * stream[bt * n * c + lane * c + h];
            }
            u[bt * c + h] = acc;
        }
    }

    // Step 4: inner encoder block (same as `EncoderLayer::forward`)
    let block_out = run_inner_block(layer, mhc, &u, layer_idx, batch, seq_len);
    // Step 5: y = block_out - u
    let mut y = vec![0.0f32; plane * c];
    for k in 0..(plane * c) {
        y[k] = block_out[k] - u[k];
    }

    // Step 6: hpost = 2 * sigmoid(a_post * (nx @ phi_post) + b_post + post_off)
    let phi_post_off = layer_idx * (n * c) * n;
    let phi_post_b_off = layer_idx * n;
    let a_post = mhc.a_post[layer_idx];
    let mut hpost = vec![0.0f32; plane * n];
    for bt in 0..plane {
        for lane in 0..n {
            let mut dot = 0.0f32;
            for k in 0..(n * c) {
                dot += nx[bt * (n * c) + k] * mhc.phi_post[phi_post_off + k * n + lane];
            }
            let v = a_post * dot + mhc.b_post[phi_post_b_off + lane] + post_off_local[lane];
            hpost[bt * n + lane] = 2.0 * sigmoid(v);
        }
    }

    // Step 7: res = nx @ phi_res  shape [B, T, 16], reshape to [B, T, 4, 4]
    let phi_res_off = layer_idx * (n * c) * (n * n);
    let phi_res_b_off = layer_idx * n * n;
    let a_res = mhc.a_res[layer_idx];
    let mut res_logits = vec![0.0f32; plane * n * n];
    for bt in 0..plane {
        for out_k in 0..(n * n) {
            let mut dot = 0.0f32;
            for k in 0..(n * c) {
                dot += nx[bt * (n * c) + k] * mhc.phi_res[phi_res_off + k * (n * n) + out_k];
            }
            let r = out_k / n;
            let c2 = out_k % n;
            res_logits[bt * (n * n) + r * n + c2] =
                a_res * dot + mhc.b_res[phi_res_b_off + r * n + c2];
        }
    }
    // Apply sinkhorn per (B, T)
    sinkhorn(&mut res_logits, batch, seq_len);
    // hres = res_logits (already doubly-stochastic-ish after sinkhorn)

    // Step 8: new_stream[bt, i, h] = Σ_j hres[bt, i, j] * xf[bt, j, h] + hpost[bt, i] * y[bt, h]
    for bt in 0..plane {
        for i in 0..n {
            for h in 0..c {
                let mut acc = 0.0f32;
                for j in 0..n {
                    acc += res_logits[bt * (n * n) + i * n + j] * stream[bt * n * c + j * c + h];
                }
                acc += hpost[bt * n + i] * y[bt * c + h];
                stream_out[bt * n * c + i * c + h] = acc;
            }
        }
    }
    if layer_idx == 0 {
        // diagnostic removed for clean test runs
    }
}

/// Inner encoder block: ZCRMSNorm → attention → ZCRMSNorm → attn_gate*skip
/// → ZCRMSNorm → HadamardMLP → +skip. Output is added back into `out`
/// (which is `u + block_residual`, the post-attention residual).
fn run_inner_block(
    layer: &EncoderLayerWeights<'_>,
    _mhc: &MhcLayer,
    u: &[f32],
    layer_idx: usize,
    batch: usize,
    seq_len: usize,
) -> Vec<f32> {
    let plane = batch * seq_len;
    let mut skip = vec![0.0f32; plane * HIDDEN];
    skip.copy_from_slice(u);

    // pre_norm = ZCRMSNorm(u, zc_in_scale[layer])
    let mut pre_norm = vec![0.0f32; plane * HIDDEN];
    zc_rms_norm_placeholder(
        u,
        &layer.zc_in_scale,
        layer_idx,
        batch,
        seq_len,
        &mut pre_norm,
    );

    // attn = GQA self_attention(pre_norm, no RoPE, no taps)
    let q_off = layer_idx * HEAD_DIM_Q;
    let k_off = layer_idx * HEAD_DIM_K;
    let mut attn = vec![0.0f32; plane * HIDDEN];
    gqa_self_attention(
        &pre_norm,
        &layer.q_proj,
        &layer.k_proj,
        &layer.v_proj,
        &layer.out_proj,
        &layer.q_norm_scale[q_off..q_off + HEAD_DIM_Q],
        &layer.k_norm_scale[k_off..k_off + HEAD_DIM_K],
        layer.attn_gate[layer_idx],
        batch,
        seq_len,
        &mut attn,
    );

    // h = skip + attn
    let mut h = vec![0.0f32; plane * HIDDEN];
    for k in 0..(plane * HIDDEN) {
        h[k] = skip[k] + attn[k];
    }
    skip.copy_from_slice(&h);

    // h_norm = ZCRMSNorm(h, post_attn_scale)
    let mut h_norm = vec![0.0f32; plane * HIDDEN];
    zc_rms_norm_placeholder(
        &h,
        &layer.post_attn_scale,
        layer_idx,
        batch,
        seq_len,
        &mut h_norm,
    );

    // mlp = hadamard_mlp(h_norm, layer_idx)
    let mut mlp = vec![0.0f32; plane * HIDDEN];
    hadamard_mlp(
        &h_norm,
        &layer.hadamard,
        layer_idx,
        batch,
        seq_len,
        &mut mlp,
    );

    // block_out = skip + mlp
    for k in 0..(plane * HIDDEN) {
        h[k] = skip[k] + mlp[k];
    }
    h
}

/// Lightweight encoder-layer view used by the mhc forward. Same fields
/// as `EncoderLayer` but without the `mhc_a_res` placeholder (real mhc
/// is computed via `MhcLayer`).
#[derive(Debug)]
pub struct EncoderLayerWeights<'a> {
    pub zc_in_scale: &'a [f32],
    pub post_attn_scale: &'a [f32],
    pub q_proj: &'a [f32],
    pub k_proj: &'a [f32],
    pub v_proj: &'a [f32],
    pub out_proj: &'a [f32],
    pub q_norm_scale: &'a [f32],
    pub k_norm_scale: &'a [f32],
    pub attn_gate: &'a [f32],
    pub hadamard: HadamardLayer,
}
