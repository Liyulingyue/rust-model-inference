// Real ZCRMSNorm — `out = (1 + scale) * x / rms(x)`.
//
// The official cactus-compute/needle definition is in
// `needle/model/architecture.py` class `ZCRMSNorm`:
//   scale = param("scale", jinit.zeros, (x.shape[-1],))
//   rms = sqrt(mean(x²) + eps)
//   out = (1 + scale) * x / rms
//
// `scale` is initialised to zeros so the first forward at init is
// identical to plain RMSNorm, but trained checkpoints learn a
// per-channel gain — and the whistle checkpoint carries a non-zero
// scale tensor.
//
// All norm scales in the safetensors are `[num_layers, hidden]`
// (per-layer, not per-head). The function picks the layer-row slice
// and applies the `(1 + scale)` affine.

use super::HIDDEN;
use super::NUM_LAYERS;

pub struct ZcRmsNormPlaceholder;

pub fn zc_rms_norm_placeholder(
    input: &[f32],
    scale: &[f32],
    layer_idx: usize,
    batch: usize,
    seq_len: usize,
    output: &mut [f32],
) {
    debug_assert_eq!(input.len(), output.len());
    debug_assert_eq!(scale.len(), NUM_LAYERS * HIDDEN);
    debug_assert!(layer_idx < NUM_LAYERS);

    let scale_row = &scale[layer_idx * HIDDEN..(layer_idx + 1) * HIDDEN];

    for b in 0..batch {
        for t in 0..seq_len {
            let base = (b * seq_len + t) * HIDDEN;
            let x = &input[base..base + HIDDEN];

            let mut sum_sq = 0.0f32;
            for &v in x {
                sum_sq += v * v;
            }
            let rms = 1.0f32 / (sum_sq / HIDDEN as f32 + 1e-6).sqrt();

            for i in 0..HIDDEN {
                output[base + i] = (1.0 + scale_row[i]) * x[i] * rms;
            }
        }
    }
}
