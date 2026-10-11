// ZCRMSNorm placeholder = plain RMSNorm (per-layer scale).
//
// Real Cactus-compute ZCRMSNorm is the next phase; we collapse it to
// RMSNorm here so the rest of the model port is shape-stable while the
// mel pipeline and Hadamard MLP / mhc / taps kernels are landing.
//
// All norm scales in the safetensors are [num_layers, hidden] — i.e.
// per-layer scale (not per-head, not per-token). The function picks
// the layer-row slice then applies RMSNorm with that scale.

use crate::ops::norm::rms_norm;
use super::HIDDEN;
use super::NUM_LAYERS;

pub struct ZcRmsNormPlaceholder;

/// RMSNorm(x) = x / sqrt(mean(x²) + eps) * scale.
///
/// `scale` is the full [NUM_LAYERS, HIDDEN] scale tensor; `layer_idx`
/// selects which row applies for this call.
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
            rms_norm(
                &input[base..base + HIDDEN],
                scale_row,
                &mut output[base..base + HIDDEN],
                1e-6,
            );
        }
    }
}
