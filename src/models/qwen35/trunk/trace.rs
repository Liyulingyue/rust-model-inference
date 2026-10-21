//! `RMI_PARITY_TRACE` checkpoints for the Qwen3.5 / `qwen35moe` trunk.
//!
//! The names mirror what the pinned llama.cpp oracle emits from
//! `src/models/qwen35.cpp` / `src/models/qwen35moe.cpp` (see
//! `tools/oracle/qwen35/README.md`), so the compare script can join the two
//! traces on `(name, layer, step)` and diff raw F32 bit patterns with no
//! tolerance.
//!
//! This module exists because `moe.rs` previously hand-rolled its own dumper
//! gated on `RMI_MOE_DUMP`: six `OpenOptions::append` files with no layer or
//! token index and no length header. With 40 layers times N tokens every record
//! interleaved into the same six files, so no dump could be attributed to a
//! `(layer, token)` pair and no comparison could falsify anything -- the same
//! failure that already invalidated one oracle result on this branch. A keyed
//! checkpoint writes one record per `(name, layer, step)` instead, so a wrong
//! number is attributable.

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
/// staging a buffer that only exists to be dumped.
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
/// The oracle writes the I32 `ffn_moe_topk` node; expert ids are small integers
/// that survive the f32 round-trip exactly, so the Rust side emits them as an
/// ordinary checkpoint and stays joinable on `(name, layer, step)`. This is the
/// join key that proves routing agrees, as opposed to merely observing that the
/// output looks plausible.
pub(crate) fn ids(name: &str, layer: usize, step: usize, values: &[u32]) {
    #[cfg(feature = "parity-trace")]
    {
        let widened: Vec<f32> = values.iter().map(|value| *value as f32).collect();
        layer_1d(name, layer, step, &widened);
    }
    #[cfg(not(feature = "parity-trace"))]
    let _ = (name, layer, step, values);
}

#[cfg(test)]
mod tests {
    /// The oracle writes `ffn_moe_topk` as I32; this side writes f32 so both
    /// traces share one join key. That only works if every expert id survives
    /// the widening exactly, which is what makes "the same experts were
    /// selected" checkable instead of merely plausible.
    #[test]
    fn expert_ids_survive_the_f32_widening_exactly() {
        let ids: Vec<u32> = (0..256).collect();
        let widened: Vec<f32> = ids.iter().map(|&value| value as f32).collect();
        assert_eq!(widened.len(), ids.len(), "no id may be dropped");
        for (got, want) in widened.iter().zip(&ids) {
            assert_eq!(
                *got as u32, *want,
                "expert id {want} must round-trip exactly"
            );
        }
    }

    /// Top-8 of 256 is the qwen35moe shape. The order is the descending-
    /// probability order, so the widened vector must keep that order rather
    /// than being sorted or de-duplicated on the way out.
    #[test]
    fn topk_order_is_preserved() {
        let chosen: Vec<u32> = vec![67, 140, 206, 3, 199, 12, 88, 251];
        let widened: Vec<f32> = chosen.iter().map(|&value| value as f32).collect();
        assert_eq!(
            widened,
            vec![67.0, 140.0, 206.0, 3.0, 199.0, 12.0, 88.0, 251.0]
        );
    }
}
