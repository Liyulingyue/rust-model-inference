//! `RMI_PARITY_TRACE` checkpoints for the Nemotron-3 Nano trunk.
//!
//! The names mirror what the upstream llama.cpp oracle emits from
//! `src/models/nemotron-h.cpp` (`build_ffn_layer`) and
//! `src/llama-graph.cpp` (`build_moe_ffn`), so the compare script can join
//! the two traces on `(name, layer, step)` and diff raw F32 bit patterns
//! with no tolerance. See `tools/oracle/nemotron3_nano/oracle.cpp` for the
//! other side.
//!
//! This module exists because the trunk previously emitted nothing: with no
//! checkpoint there was no way to tell a wrong routing weight from a wrong
//! expert projection, and "the output looks plausible" is not a check. A
//! keyed checkpoint writes one record per `(name, layer, step)`, so a wrong
//! number is attributable to the layer and token that produced it.
//!
//! `step` is the **position** of the token in the sequence, not the call
//! index. The oracle's `trace.step` counts positions the same way, so the two
//! join without either side having to know how the other chunks its batches.

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

/// Emit a whole-model checkpoint (embedding, final norm, logits). The oracle
/// writes layer `-1` for these; `parity_trace` writes `null`, and the compare
/// script normalises the two.
pub(crate) fn global(name: &str, step: usize, values: &[f32]) {
    checkpoint(name, None, step, &[values.len()], values);
}

/// Emit the integer top-k expert selection.
///
/// The oracle writes the I32 `ffn_moe_topk` node; expert ids are small
/// integers that survive the f32 round-trip exactly, so the Rust side emits
/// them as an ordinary checkpoint and stays joinable on `(name, layer, step)`.
/// This is the join key that proves routing agrees, as opposed to merely
/// observing that the output looks plausible.
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
    /// selected" checkable instead of merely plausible. 128 experts is the
    /// 30B's count, so the range is the one that matters.
    #[test]
    fn expert_ids_survive_the_f32_widening_exactly() {
        let ids: Vec<u32> = (0..128).collect();
        let widened: Vec<f32> = ids.iter().map(|&value| value as f32).collect();
        assert_eq!(widened.len(), ids.len(), "no id may be dropped");
        for (got, want) in widened.iter().zip(&ids) {
            assert_eq!(
                *got as u32, *want,
                "expert id {want} must round-trip exactly"
            );
        }
    }

    /// Top-6 of 128 is the 30B's shape. The order is the descending-selection
    /// order, so the widened vector must keep that order rather than being
    /// sorted or de-duplicated on the way out.
    #[test]
    fn topk_order_is_preserved() {
        let chosen: Vec<u32> = vec![117, 3, 88, 0, 64, 12];
        let widened: Vec<f32> = chosen.iter().map(|&value| value as f32).collect();
        assert_eq!(widened, vec![117.0, 3.0, 88.0, 0.0, 64.0, 12.0]);
    }
}
