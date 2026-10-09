//! Helpers shared by the diffusion pipelines.
//!
//! These are the tensor-contract checks that every pipeline performs when
//! loading its components. They were duplicated verbatim across `auk`,
//! `ernie_image` and `z_image`, which meant a change to the contract had to be
//! made in three places and could silently diverge.
//!
//! Only genuinely identical code lives here. Where a pipeline has a richer
//! diagnostic or different behaviour -- Z-Image's `require_finite` reports
//! nan/inf counts and the first bad index, and its `resize_zeroed` avoids
//! clearing a buffer it can reuse -- it keeps its own version and calls the
//! shared predicate instead.

use crate::core::tensor::GGMLType;
use crate::core::tensor::TensorSource;

/// Whether every element is finite. NaN and infinities are both rejected.
pub(crate) fn all_finite(values: &[f32]) -> bool {
    values.iter().all(|v| v.is_finite())
}

/// Assert that a tensor exists with the expected shape, dtype and byte length.
///
/// The dtype and byte-length checks are what catch a GGUF that was quantized
/// differently from the one a pipeline was validated against, which otherwise
/// shows up much later as an out-of-bounds read.
pub(crate) fn require_tensor(
    source: &dyn TensorSource,
    name: &str,
    dims: &[u64],
    ggml_type: GGMLType,
) -> Result<(), String> {
    let info = source
        .tensor_info(name)
        .ok_or_else(|| format!("Missing tensor: {name}"))?;
    if info.dims != dims {
        return Err(format!("Invalid {name} dimensions"));
    }
    if info.ggml_type != ggml_type {
        return Err(format!(
            "Invalid {name} type: expected {ggml_type:?}, got {:?}",
            info.ggml_type
        ));
    }
    let expected = usize::try_from(
        info.checked_nbytes()
            .ok_or_else(|| format!("Tensor byte size does not fit usize: {name}"))?,
    )
    .map_err(|_| format!("Tensor byte size does not fit usize: {name}"))?;
    let bytes = source
        .tensor_slice(name)
        .ok_or_else(|| format!("Missing tensor data: {name}"))?;
    if bytes.len() != expected {
        return Err(format!("Invalid {name} byte length"));
    }
    Ok(())
}

/// Reject a buffer containing NaN or infinities.
pub(crate) fn require_finite(values: &[f32], name: &str) -> Result<(), String> {
    if all_finite(values) {
        Ok(())
    } else {
        Err(format!("Non-finite {name}"))
    }
}

/// Resize `dst` to `len` zeros, reporting allocation failures by `name`.
///
/// This always clears first, so a buffer that shrinks discards its tail.
pub(crate) fn resize_zeroed(dst: &mut Vec<f32>, len: usize, name: &str) -> Result<(), String> {
    dst.clear();
    dst.try_reserve_exact(len)
        .map_err(|e| format!("Failed to allocate {name}: {e}"))?;
    dst.resize(len, 0.0);
    Ok(())
}
