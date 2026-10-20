//! `RMI_PARITY_TRACE` checkpoints for the Mellum trunk.
//!
//! The names mirror what the pinned llama.cpp oracle emits from
//! `src/models/mellum.cpp` (`tools/oracle/mellum/README.md`), so a compare
//! script can join the two traces on `(name, layer, step)` and diff raw F32
//! bit patterns.

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

/// Emit the whole-model checkpoints (embedding, final norm, logits).
pub(crate) fn global(name: &str, step: usize, values: &[f32]) {
    checkpoint(name, None, step, &[values.len()], values);
}

/// Emit the integer top-k expert selection as an f32 checkpoint.
///
/// `usize_values` has no layer/step fields, so it cannot join against the
/// oracle's per-layer `int_ids`; expert ids are small integers and survive
/// the f32 round-trip exactly, so emitting them as a normal checkpoint keeps
/// `(name, layer, step)` joinable.
pub(crate) fn ids(name: &str, layer: usize, step: usize, values: &[u32]) {
    #[cfg(feature = "parity-trace")]
    {
        let widened: Vec<f32> = values.iter().map(|value| *value as f32).collect();
        layer_1d(name, layer, step, &widened);
    }
    #[cfg(not(feature = "parity-trace"))]
    let _ = (name, layer, step, values);
}

/// Emit a two-dimensional checkpoint as a flat row-major buffer.
pub(crate) fn layer_2d(
    name: &str,
    layer: usize,
    step: usize,
    rows: usize,
    cols: usize,
    values: &[f32],
) {
    checkpoint(name, Some(layer), step, &[cols, rows], values);
}
