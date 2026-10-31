//! LFM3 checkpoints for the CPU reference runner in tools/oracle/lfm3.

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

pub(crate) fn layer_2d(
    name: &str,
    layer: usize,
    step: usize,
    dim: usize,
    heads: usize,
    values: &[f32],
) {
    checkpoint(name, Some(layer), step, &[dim, heads], values);
}

/// Record exact prompt / generated token IDs alongside numerical checkpoints.
pub(crate) fn tokens(name: &str, values: &[u32]) {
    #[cfg(feature = "parity-trace")]
    {
        let _ = crate::parity_trace::token_ids(name, values);
    }
    #[cfg(not(feature = "parity-trace"))]
    let _ = (name, values);
}
