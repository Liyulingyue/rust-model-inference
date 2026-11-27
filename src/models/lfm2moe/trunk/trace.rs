//! `RMI_PARITY_TRACE` checkpoints for the LFM2-MoE trunk.
//!
//! The names mirror what the pinned llama.cpp oracle emits from
//! `src/models/lfm2.cpp` (see `tools/oracle/lfm2moe/README.md`), so the compare
//! script can join the two traces on `(name, layer, step)` and diff raw F32
//! bit patterns with no tolerance.

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

/// True when this checkpoint would actually be written. Lets a caller skip
/// building a staging buffer (e.g. the per-expert MoE `down` slab) that only
/// exists to be dumped.
pub(crate) fn trace_enabled(name: &str) -> bool {
    #[cfg(feature = "parity-trace")]
    {
        crate::parity_trace::enabled(name)
    }
    #[cfg(not(feature = "parity-trace"))]
    {
        let _ = name;
        false
    }
}

/// Emit a per-layer checkpoint whose shape is one-dimensional.
pub(crate) fn layer_1d(name: &str, layer: usize, step: usize, values: &[f32]) {
    checkpoint(name, Some(layer), step, &[values.len()], values);
}

/// Emit a whole-model checkpoint (embedding, final norm, logits).
pub(crate) fn global(name: &str, step: usize, values: &[f32]) {
    checkpoint(name, None, step, &[values.len()], values);
}

/// Emit the integer top-k expert selection.
///
/// The oracle side writes the I32 `ffn_moe_topk` node as `int_ids`; expert ids
/// are small integers that survive the f32 round-trip exactly, so the Rust side
/// emits them as an ordinary checkpoint and stays joinable on
/// `(name, layer, step)`.
pub(crate) fn ids(name: &str, layer: usize, step: usize, values: &[u32]) {
    #[cfg(feature = "parity-trace")]
    {
        let widened: Vec<f32> = values.iter().map(|value| *value as f32).collect();
        layer_1d(name, layer, step, &widened);
    }
    #[cfg(not(feature = "parity-trace"))]
    let _ = (name, layer, step, values);
}
