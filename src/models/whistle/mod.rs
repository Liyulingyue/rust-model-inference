// Whistle (Cactus-Compute) ASR architecture.
//
// Placeholder port (Phase 1): only validates shapes and basic forward
// parity against `tools/converter/whistle/probe/oracle.py`.
//
// Placeholders (will be replaced with real Cactus-compute ops once we
// verify them against an external numerical reference):
//   - ZCRMSNorm        → plain RMSNorm
//   - mixed_hadamard_skip → identity (phi_pre/post/res all folded to 0)
//   - taps             → [0, 1, 0] (no-op causal FIR)
//   - conv stem        → random init by the caller; encoder/decoder block
//                        are validated against an Oracle that injects the
//                        same hidden-dim placeholder.
//
// Real implementations come in Phase 2 (stem), Phase 3 (ZCRMSNorm),
// Phase 4 (mhc + taps).

pub mod attention;
pub mod hadamard;
pub mod mhc;
pub mod norm;
pub mod stack;
pub mod stem;

pub use attention::{
    gqa_cross_attention, gqa_self_attention, gqa_self_attention_with_taps_and_rope,
};
pub use hadamard::hadamard_mlp;
pub use mhc::{forward_one, load_mhc_layer, EncoderLayerWeights, MhcLayer, MHC_LANES};
pub use norm::{zc_rms_norm_placeholder, ZcRmsNormPlaceholder};
pub use stack::{final_rms_norm, load_stack_layer, precompute_rope, StackLayer};
pub use stem::placeholder_stem_output;

use crate::core::tensor::{load_f32_tensor, TensorSource};
use crate::models::whistle::hadamard::HadamardLayer;

/// Load a per-layer kernel slice, returning `[HIDDEN, OUT]` row-major.
/// Most whistle kernels are stored as `[NUM_LAYERS, HIDDEN, OUT]`; this
/// helper copies the requested layer row out into a fresh vector.
fn load_layer_kernel_2d(
    source: &dyn TensorSource,
    name: &str,
    layer_idx: usize,
    inner: &[u64],
) -> Vec<f32> {
    let outer = NUM_LAYERS as u64;
    let full_dims = std::iter::once(outer)
        .chain(inner.iter().copied())
        .collect::<Vec<_>>();
    let raw = load_f32_tensor(source, name, &full_dims).expect(name);
    // `inner` is `[HIDDEN, OUT]` for this helper.
    let out_dim: usize = inner[1..].iter().product::<u64>().try_into().unwrap();
    let plane = HIDDEN * out_dim;
    let mut slice = vec![0.0f32; plane];
    let layer_off = layer_idx * plane;
    for i in 0..HIDDEN {
        for o in 0..out_dim {
            slice[i * out_dim + o] = raw[layer_off + i * out_dim + o];
        }
    }
    slice
}

/// Like `load_layer_kernel_2d` but for kernels stored as
/// `[NUM_LAYERS, A, B, C]` (used by Hadamard's w1a/w1b etc).
fn load_layer_kernel_nd(
    source: &dyn TensorSource,
    name: &str,
    layer_idx: usize,
    inner: &[u64],
) -> Vec<f32> {
    let outer = NUM_LAYERS as u64;
    let full_dims = std::iter::once(outer)
        .chain(inner.iter().copied())
        .collect::<Vec<_>>();
    let raw = load_f32_tensor(source, name, &full_dims).expect(name);
    let inner_count: usize = inner.iter().product::<u64>().try_into().unwrap();
    raw[layer_idx * inner_count..(layer_idx + 1) * inner_count].to_vec()
}

/// Architectural constants reverse-engineered from
/// `whistle.safetensors` (116 tensors, 221 MB, F32).
pub const HIDDEN: usize = 512;
pub const NUM_LAYERS: usize = 8;
pub const NUM_HEADS: usize = 8;
pub const NUM_KV_HEADS: usize = 2;
pub const HEAD_DIM_Q: usize = 48;
pub const HEAD_DIM_K: usize = 48;
pub const HEAD_DIM_V: usize = 64;
pub const TEXT_VOCAB: usize = 8199;

/// One full encoder block layer, holding the per-layer weight tensors
/// needed for: zc_rms_in → attn → post_attn_norm → hadamard_mlp + skip.
#[derive(Debug, Clone)]
pub struct EncoderLayer {
    pub zc_in_scale: Vec<f32>,     // [NUM_LAYERS, HIDDEN]
    pub post_attn_scale: Vec<f32>, // [NUM_LAYERS, HIDDEN]
    pub pre_hada_scale: Vec<f32>,  // [NUM_LAYERS, HIDDEN] (unused placeholder)
    pub q_proj: Vec<f32>,          // [HIDDEN, NUM_HEADS * HEAD_DIM_Q]
    pub k_proj: Vec<f32>,          // [HIDDEN, NUM_KV_HEADS * HEAD_DIM_K]
    pub v_proj: Vec<f32>,          // [HIDDEN, NUM_KV_HEADS * HEAD_DIM_V]
    pub out_proj: Vec<f32>,        // [HIDDEN, HIDDEN]
    pub q_norm_scale: Vec<f32>,    // [NUM_LAYERS, HEAD_DIM_Q] (per-layer)
    pub k_norm_scale: Vec<f32>,    // [NUM_LAYERS, HEAD_DIM_K] (per-layer)
    pub attn_gate: Vec<f32>,       // [NUM_LAYERS]
    pub hadamard: HadamardLayer,
}

/// Load a single encoder layer's weights from the safetensors/GGUF source.
pub fn load_encoder_layer(source: &dyn TensorSource, layer_idx: usize) -> EncoderLayer {
    let p = "encoder/layers/block";
    EncoderLayer {
        zc_in_scale: load_f32_tensor(
            source,
            &format!("{p}/ZCRMSNorm_0/scale"),
            &[NUM_LAYERS as u64, HIDDEN as u64],
        )
        .expect("zc_in_scale"),
        post_attn_scale: load_f32_tensor(
            source,
            &format!("{p}/post_attn_norm/scale"),
            &[NUM_LAYERS as u64, HIDDEN as u64],
        )
        .expect("post_attn_scale"),
        pre_hada_scale: load_f32_tensor(
            source,
            &format!("{p}/pre_hada_norm/scale"),
            &[NUM_LAYERS as u64, HIDDEN as u64],
        )
        .expect("pre_hada_scale"),
        q_proj: load_layer_kernel_2d(
            source,
            &format!("{p}/self_attn/q_proj/kernel"),
            layer_idx,
            &[HIDDEN as u64, (NUM_HEADS * HEAD_DIM_Q) as u64],
        ),
        k_proj: load_layer_kernel_2d(
            source,
            &format!("{p}/self_attn/k_proj/kernel"),
            layer_idx,
            &[HIDDEN as u64, (NUM_KV_HEADS * HEAD_DIM_K) as u64],
        ),
        v_proj: load_layer_kernel_2d(
            source,
            &format!("{p}/self_attn/v_proj/kernel"),
            layer_idx,
            &[HIDDEN as u64, (NUM_KV_HEADS * HEAD_DIM_V) as u64],
        ),
        out_proj: load_layer_kernel_2d(
            source,
            &format!("{p}/self_attn/out_proj/kernel"),
            layer_idx,
            &[HIDDEN as u64, HIDDEN as u64],
        ),
        q_norm_scale: load_f32_tensor(
            source,
            &format!("{p}/self_attn/q_norm/scale"),
            &[NUM_LAYERS as u64, HEAD_DIM_Q as u64],
        )
        .expect("q_norm_scale"),
        k_norm_scale: load_f32_tensor(
            source,
            &format!("{p}/self_attn/k_norm/scale"),
            &[NUM_LAYERS as u64, HEAD_DIM_K as u64],
        )
        .expect("k_norm_scale"),
        attn_gate: load_f32_tensor(source, &format!("{p}/attn_gate"), &[NUM_LAYERS as u64])
            .expect("attn_gate"),
        hadamard: HadamardLayer {
            w1a: load_layer_kernel_nd(
                source,
                &format!("{p}/hadamard_mlp/w1a"),
                layer_idx,
                &[16, 16],
            ),
            w1b: load_layer_kernel_nd(
                source,
                &format!("{p}/hadamard_mlp/w1b"),
                layer_idx,
                &[32, 32],
            ),
            w2a: load_layer_kernel_nd(
                source,
                &format!("{p}/hadamard_mlp/w2a"),
                layer_idx,
                &[16, 16],
            ),
            w2b: load_layer_kernel_nd(
                source,
                &format!("{p}/hadamard_mlp/w2b"),
                layer_idx,
                &[32, 32],
            ),
            w3a: load_layer_kernel_nd(
                source,
                &format!("{p}/hadamard_mlp/w3a"),
                layer_idx,
                &[16, 16],
            ),
            w3b: load_layer_kernel_nd(
                source,
                &format!("{p}/hadamard_mlp/w3b"),
                layer_idx,
                &[32, 32],
            ),
            d1: load_layer_kernel_nd(
                source,
                &format!("{p}/hadamard_mlp/d1"),
                layer_idx,
                &[HIDDEN as u64],
            ),
            d2: load_layer_kernel_nd(
                source,
                &format!("{p}/hadamard_mlp/d2"),
                layer_idx,
                &[HIDDEN as u64],
            ),
            d3: load_layer_kernel_nd(
                source,
                &format!("{p}/hadamard_mlp/d3"),
                layer_idx,
                &[HIDDEN as u64],
            ),
            d4: load_layer_kernel_nd(
                source,
                &format!("{p}/hadamard_mlp/d4"),
                layer_idx,
                &[HIDDEN as u64],
            ),
            b2: load_layer_kernel_nd(
                source,
                &format!("{p}/hadamard_mlp/b2"),
                layer_idx,
                &[HIDDEN as u64],
            ),
            cond_u: load_layer_kernel_nd(
                source,
                &format!("{p}/hadamard_mlp/cond_u"),
                layer_idx,
                &[8, HIDDEN as u64],
            ),
            cond_v: load_layer_kernel_nd(
                source,
                &format!("{p}/hadamard_mlp/cond_v"),
                layer_idx,
                &[HIDDEN as u64, 8],
            ),
        },
    }
}

impl EncoderLayer {
    /// Forward one encoder block on `[B, T, HIDDEN]` input, returning the
    /// same-shape output. Placeholder ZCRMSNorm/mhc/taps are applied.
    pub fn forward(
        &self,
        x: &[f32],
        layer_idx: usize,
        batch: usize,
        seq_len: usize,
        out: &mut [f32],
    ) {
        debug_assert_eq!(x.len(), batch * seq_len * HIDDEN);
        debug_assert_eq!(out.len(), batch * seq_len * HIDDEN);

        let mut normed = vec![0.0f32; batch * seq_len * HIDDEN];
        zc_rms_norm_placeholder(x, &self.zc_in_scale, layer_idx, batch, seq_len, &mut normed);

        let q_scale_off = layer_idx * HEAD_DIM_Q;
        let k_scale_off = layer_idx * HEAD_DIM_K;
        let mut attn_out = vec![0.0f32; batch * seq_len * HIDDEN];
        gqa_self_attention(
            &normed,
            &self.q_proj,
            &self.k_proj,
            &self.v_proj,
            &self.out_proj,
            &self.q_norm_scale[q_scale_off..q_scale_off + HEAD_DIM_Q],
            &self.k_norm_scale[k_scale_off..k_scale_off + HEAD_DIM_K],
            self.attn_gate[layer_idx],
            batch,
            seq_len,
            &mut attn_out,
        );

        // residual add: out = x + attn_out
        for i in 0..(batch * seq_len * HIDDEN) {
            out[i] = x[i] + attn_out[i];
        }

        // post_attn_norm + hadamard_mlp + residual
        let mut h = vec![0.0f32; batch * seq_len * HIDDEN];
        zc_rms_norm_placeholder(
            out,
            &self.post_attn_scale,
            layer_idx,
            batch,
            seq_len,
            &mut h,
        );

        let mut mlp = vec![0.0f32; batch * seq_len * HIDDEN];
        hadamard_mlp(&h, &self.hadamard, layer_idx, batch, seq_len, &mut mlp);

        // out += mlp (skip-style residual is taken by the surrounding mhc;
        // this inner-block path is only used by `mhc::run_inner_block`).
        for i in 0..(batch * seq_len * HIDDEN) {
            out[i] += mlp[i];
        }
    }
}
