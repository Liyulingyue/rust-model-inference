// Text-decoder stack layer for Whistle (single 1-layer decoder over
// the encoder's audio hidden states).
//
// Architecture (per the Oracle placeholder):
//   zc_rms_in → self_attn (GQA 4:1) → residual
//   cross_norm → cross_attn (full MHA, kv=encoder out) → residual
//   final_norm → logits = text_embed @ embedding.weight.T
//
// Taps / Hadamard MLP / mhc skip on the stack are not part of the
// placeholder parity target; they are flagged as TODO and will land
// alongside the encoder's Phase 3 / Phase 4 lifts.

use crate::core::tensor::{load_f32_tensor, TensorSource};
use crate::models::whistle::attention::{
    gqa_cross_attention, gqa_self_attention,
};
use crate::models::whistle::hadamard::HadamardLayer;
use crate::models::whistle::norm::zc_rms_norm_placeholder;

use super::{
    HIDDEN, NUM_HEADS, NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM_K, HEAD_DIM_Q,
    HEAD_DIM_V,
};

/// Stack layer weights (single layer for the placeholder decoder).
#[derive(Debug)]
pub struct StackLayer {
    pub zc_in_scale: Vec<f32>,     // [NUM_LAYERS, HIDDEN]
    pub cross_norm_scale: Vec<f32>, // [NUM_LAYERS, HIDDEN]
    pub self_q_proj: Vec<f32>,
    pub self_k_proj: Vec<f32>,
    pub self_v_proj: Vec<f32>,
    pub self_out_proj: Vec<f32>,
    pub self_q_norm_scale: Vec<f32>, // [NUM_LAYERS, HEAD_DIM_Q]
    pub self_k_norm_scale: Vec<f32>, // [NUM_LAYERS, HEAD_DIM_K]
    pub self_attn_gate: Vec<f32>,    // [NUM_LAYERS]
    pub cross_q_proj: Vec<f32>,
    pub cross_k_proj: Vec<f32>,
    pub cross_v_proj: Vec<f32>,
    pub cross_out_proj: Vec<f32>,
    pub cross_q_norm_scale: Vec<f32>,
    pub cross_k_norm_scale: Vec<f32>,
    pub cross_gate: Vec<f32>,
}

fn load_kernel_2d(
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

pub fn load_stack_layer(source: &dyn TensorSource, layer_idx: usize) -> StackLayer {
    let p = "stack/layers/block";
    StackLayer {
        zc_in_scale: load_f32_tensor(
            source,
            &format!("{p}/ZCRMSNorm_0/scale"),
            &[NUM_LAYERS as u64, HIDDEN as u64],
        )
        .expect("zc_in_scale"),
        cross_norm_scale: load_f32_tensor(
            source,
            &format!("{p}/cross_norm/scale"),
            &[NUM_LAYERS as u64, HIDDEN as u64],
        )
        .expect("cross_norm_scale"),
        self_q_proj: load_kernel_2d(
            source,
            &format!("{p}/self_attn/q_proj/kernel"),
            layer_idx,
            &[HIDDEN as u64, (NUM_HEADS * HEAD_DIM_Q) as u64],
        ),
        self_k_proj: load_kernel_2d(
            source,
            &format!("{p}/self_attn/k_proj/kernel"),
            layer_idx,
            &[HIDDEN as u64, (NUM_KV_HEADS * HEAD_DIM_K) as u64],
        ),
        self_v_proj: load_kernel_2d(
            source,
            &format!("{p}/self_attn/v_proj/kernel"),
            layer_idx,
            &[HIDDEN as u64, (NUM_KV_HEADS * HEAD_DIM_V) as u64],
        ),
        self_out_proj: load_kernel_2d(
            source,
            &format!("{p}/self_attn/out_proj/kernel"),
            layer_idx,
            &[HIDDEN as u64, HIDDEN as u64],
        ),
        self_q_norm_scale: load_f32_tensor(
            source,
            &format!("{p}/self_attn/q_norm/scale"),
            &[NUM_LAYERS as u64, HEAD_DIM_Q as u64],
        )
        .expect("self_q_norm_scale"),
        self_k_norm_scale: load_f32_tensor(
            source,
            &format!("{p}/self_attn/k_norm/scale"),
            &[NUM_LAYERS as u64, HEAD_DIM_K as u64],
        )
        .expect("self_k_norm_scale"),
        self_attn_gate: load_f32_tensor(
            source,
            &format!("{p}/attn_gate"),
            &[NUM_LAYERS as u64],
        )
        .expect("attn_gate"),
        cross_q_proj: load_kernel_2d(
            source,
            &format!("{p}/cross_attn/q_proj/kernel"),
            layer_idx,
            &[HIDDEN as u64, (NUM_HEADS * HEAD_DIM_Q) as u64],
        ),
        cross_k_proj: load_kernel_2d(
            source,
            &format!("{p}/cross_attn/k_proj/kernel"),
            layer_idx,
            &[HIDDEN as u64, (NUM_HEADS * HEAD_DIM_K) as u64],
        ),
        cross_v_proj: load_kernel_2d(
            source,
            &format!("{p}/cross_attn/v_proj/kernel"),
            layer_idx,
            &[HIDDEN as u64, (NUM_HEADS * HEAD_DIM_V) as u64],
        ),
        cross_out_proj: load_kernel_2d(
            source,
            &format!("{p}/cross_attn/out_proj/kernel"),
            layer_idx,
            &[HIDDEN as u64, HIDDEN as u64],
        ),
        cross_q_norm_scale: load_f32_tensor(
            source,
            &format!("{p}/cross_attn/q_norm/scale"),
            &[NUM_LAYERS as u64, HEAD_DIM_Q as u64],
        )
        .expect("cross_q_norm_scale"),
        cross_k_norm_scale: load_f32_tensor(
            source,
            &format!("{p}/cross_attn/k_norm/scale"),
            &[NUM_LAYERS as u64, HEAD_DIM_K as u64],
        )
        .expect("cross_k_norm_scale"),
        cross_gate: load_f32_tensor(source, &format!("{p}/cross_gate"), &[NUM_LAYERS as u64])
            .expect("cross_gate"),
    }
}

impl StackLayer {
    /// Forward one stack block. `text_emb`: [B, S_text, HIDDEN],
    /// `enc_out`: [B, S_audio, HIDDEN] from the encoder.
    pub fn forward(
        &self,
        text_emb: &[f32],
        enc_out: &[f32],
        layer_idx: usize,
        batch: usize,
        q_seq: usize,
        kv_seq: usize,
        out: &mut [f32],
    ) {
        debug_assert_eq!(text_emb.len(), batch * q_seq * HIDDEN);
        debug_assert_eq!(enc_out.len(), batch * kv_seq * HIDDEN);
        debug_assert_eq!(out.len(), batch * q_seq * HIDDEN);

        // 1) ZCRMSNorm + self-attn
        let mut normed = vec![0.0f32; batch * q_seq * HIDDEN];
        zc_rms_norm_placeholder(
            text_emb,
            &self.zc_in_scale,
            layer_idx,
            batch,
            q_seq,
            &mut normed,
        );

        let q_scale_off = layer_idx * HEAD_DIM_Q;
        let k_scale_off = layer_idx * HEAD_DIM_K;
        let mut sa = vec![0.0f32; batch * q_seq * HIDDEN];
        gqa_self_attention(
            &normed,
            &self.self_q_proj,
            &self.self_k_proj,
            &self.self_v_proj,
            &self.self_out_proj,
            &self.self_q_norm_scale[q_scale_off..q_scale_off + HEAD_DIM_Q],
            &self.self_k_norm_scale[k_scale_off..k_scale_off + HEAD_DIM_K],
            self.self_attn_gate[layer_idx],
            batch,
            q_seq,
            &mut sa,
        );
        // residual
        for i in 0..out.len() {
            out[i] = text_emb[i] + sa[i];
        }

        // 2) cross_norm + cross-attn (kv from encoder out)
        let mut pre_cross = vec![0.0f32; batch * q_seq * HIDDEN];
        zc_rms_norm_placeholder(
            out,
            &self.cross_norm_scale,
            layer_idx,
            batch,
            q_seq,
            &mut pre_cross,
        );
        let mut cross = vec![0.0f32; batch * q_seq * HIDDEN];
        gqa_cross_attention(
            &pre_cross,
            enc_out,
            &self.cross_q_proj,
            &self.cross_k_proj,
            &self.cross_v_proj,
            &self.cross_out_proj,
            &self.cross_q_norm_scale[q_scale_off..q_scale_off + HEAD_DIM_Q],
            &self.cross_k_norm_scale[k_scale_off..k_scale_off + HEAD_DIM_K],
            self.cross_gate[layer_idx],
            batch,
            q_seq,
            kv_seq,
            &mut cross,
        );
        for i in 0..out.len() {
            out[i] += cross[i];
        }
    }
}

/// Walk `vec` through per-token RMSNorm with a shared `[HIDDEN]` scale.
pub fn final_rms_norm(input: &[f32], scale: &[f32], batch: usize, seq_len: usize, out: &mut [f32]) {
    use crate::ops::norm::rms_norm;
    debug_assert_eq!(scale.len(), HIDDEN);
    for b in 0..batch {
        for t in 0..seq_len {
            let base = (b * seq_len + t) * HIDDEN;
            rms_norm(
                &input[base..base + HIDDEN],
                scale,
                &mut out[base..base + HIDDEN],
                1e-6,
            );
        }
    }
}

// Keep `HadamardLayer` reference active for follow-on phases.
#[allow(dead_code)]
fn _keep(_: &HadamardLayer) {}