//! `RMI_PARITY_TRACE` checkpoints for the dense LFM2 trunk.
//!
//! The names and shapes mirror what the pinned llama.cpp oracle emits from
//! `src/models/lfm2.cpp` (see `tools/oracle/lfm2moe/README.md`), so
//! `tools/oracle/lfm2moe/check_scalar.py` can join the two manifests on
//! `(name, layer, step)` and diff raw F32 bit patterns with no tolerance.
//!
//! The oracle labels 24 tensors per step. For a 14-layer LFM2.5-230M that is
//! `attn_norm`, `post_attn_residual`, `ffn_norm`, `ffn_up`, `ffn_gate`,
//! `ffn_swiglu`, `ffn_down`, `l_out` on every layer; `conv_in_proj`,
//! `conv_conv`, `conv_out_proj` on the shortconv layers; `q_proj`, `k_proj`,
//! `v_proj`, `q_norm`, `k_norm`, `q_rope`, `k_rope`, `attn_scores`,
//! `attn_probs`, `attn_values`, `attn_proj` on the attention layers; and the
//! `result_norm` / `result_output` globals.
//!
//! Only `Lfm2Session`'s free-function path is instrumented. `Lfm25Session` is a
//! near-duplicate trunk that `tests/lfm2_dense_trunk_agreement.rs` proves
//! produces bit-identical logits, so tracing one of them localises a divergence
//! for both.

/// Emit one checkpoint. A no-op unless the crate is built with the
/// `parity-trace` feature and `RMI_PARITY_TRACE` is set.
pub(crate) fn checkpoint(
    name: &str,
    layer: Option<usize>,
    step: usize,
    shape: &[usize],
    values: &[f32],
) {
    #[cfg(feature = "parity-trace")]
    crate::parity_trace::report(crate::parity_trace::checkpoint_at(
        name,
        layer,
        Some(step),
        shape,
        values,
    ));
    #[cfg(not(feature = "parity-trace"))]
    let _ = (name, layer, step, shape, values);
}

/// Emit a per-layer checkpoint whose shape is one-dimensional.
pub(crate) fn layer_1d(name: &str, layer: usize, step: usize, values: &[f32]) {
    checkpoint(name, Some(layer), step, &[values.len()], values);
}

/// Emit a whole-model checkpoint (final norm, logits).
pub(crate) fn global(name: &str, step: usize, values: &[f32]) {
    checkpoint(name, None, step, &[values.len()], values);
}

/// Emit a per-layer checkpoint with an explicit two-dimensional shape.
///
/// ggml writes projections as `[out_features, n_embd_head]`, so the oracle
/// records e.g. `q_proj` with shape `[64, 16]` rather than `[1024]`. Keeping
/// the same rank lets the compare script validate the shape too.
pub(crate) fn layer_2d(
    name: &str,
    layer: usize,
    step: usize,
    rows: usize,
    cols: usize,
    values: &[f32],
) {
    checkpoint(name, Some(layer), step, &[rows, cols], values);
}
